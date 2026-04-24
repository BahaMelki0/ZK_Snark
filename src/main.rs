// ZK-SNARK / Lattice Polynomial Commitment — clean implementation.
//
// Based on: "Polynomial Commitments from Lattices" (Cini, Malavolta, Nguyen, Wee – ePrint 2024/281).
// Implements:
//   • Section 4 – tree-based SIS commitment        (commit / verify_commit)
//   • Section 5, Figure 6 – full recursive protocol (run_protocol)
//       b₀: (I_κ ⊗ A) · y_i = t^{(i)}          (SIS commitment check)
//       b₁: ‖y_i‖_∞ ≤ β_i                       (norm bound)
//       b₂: (I_{κn} ⊗ x[ℓ-1-i]) · v_i = u^{(i)} (evaluation check)
//
// All arithmetic is i64 mod q.  No finite-field library needed.

use rand::{thread_rng, Rng};

// ============================================================
// 1.  Ring arithmetic  Z_q
// ============================================================

fn modq(x: i64, q: i64) -> i64 {
    ((x % q) + q) % q
}

// ============================================================
// 2.  Matrix / vector helpers
// ============================================================

type Vec1 = Vec<i64>;
type Mat  = Vec<Vec<i64>>; // row-major: m[row][col]

fn rows(m: &Mat) -> usize { m.len() }
fn cols(m: &Mat) -> usize { m.first().map(|r| r.len()).unwrap_or(0) }

fn mat_zero(r: usize, c: usize) -> Mat { vec![vec![0i64; c]; r] }

fn eye(n: usize) -> Mat {
    let mut m = mat_zero(n, n);
    for i in 0..n { m[i][i] = 1; }
    m
}

fn transpose(m: &Mat) -> Mat {
    let (r, c) = (rows(m), cols(m));
    let mut t = mat_zero(c, r);
    for i in 0..r { for j in 0..c { t[j][i] = m[i][j]; } }
    t
}

/// Matrix–vector product.  q == 0 → integer arithmetic (no mod).
fn mv(m: &Mat, v: &Vec1, q: i64) -> Vec1 {
    let (r, c) = (rows(m), cols(m));
    assert_eq!(c, v.len(), "mv: {}×{} · len={}", r, c, v.len());
    (0..r).map(|i| {
        let s: i64 = (0..c).map(|j| m[i][j] * v[j]).sum();
        if q == 0 { s } else { modq(s, q) }
    }).collect()
}

/// A ⊗ B  (Kronecker product, pure integer — no mod)
fn kron(a: &Mat, b: &Mat) -> Mat {
    let (ar, ac) = (rows(a), cols(a));
    let (br, bc) = (rows(b), cols(b));
    let mut out = mat_zero(ar * br, ac * bc);
    for i in 0..ar { for j in 0..ac {
        for k in 0..br { for l in 0..bc {
            out[i * br + k][j * bc + l] = a[i][j] * b[k][l];
        }}
    }}
    out
}

/// ℓ∞ norm
fn linf(v: &Vec1) -> i64 {
    v.iter().map(|&x| x.abs()).max().unwrap_or(0)
}

// ============================================================
// 3.  Gadget matrix  G_{m, α, δ}  — shape  m × (m·α)
//
//   Row i has δ⁰, δ¹, …, δ^{α-1} at columns i·α … (i+1)·α−1.
//
// G^{−1} : Z_q^m → {0,…,δ-1}^{m·α}
//   Little-endian base-δ decomposition of each element.
//   Satisfies  G · G^{−1}(v) = v  (mod q).
// ============================================================

fn gadget(m: usize, alpha: usize, delta: i64) -> Mat {
    let mut g = mat_zero(m, m * alpha);
    for i in 0..m {
        let mut pw = 1i64;
        for k in 0..alpha {
            g[i][i * alpha + k] = pw;
            pw *= delta;
        }
    }
    g
}

/// G^{−1}: base-δ decomposition, output length = input.len() · α.
fn g_inv(v: &Vec1, q: i64, delta: i64, alpha: usize) -> Vec1 {
    v.iter().flat_map(|&x| {
        let mut rem = modq(x, q) as u64;
        let d = delta as u64;
        (0..alpha).map(move |_| {
            let b = (rem % d) as i64;
            rem /= d;
            b
        })
    }).collect()
}

// ============================================================
// 4.  Scheme parameters
//
//   A   ∈ Z_q^{n × r·n·α}        (SIS matrix)
//   x   = [x_0,…,x_{ℓ-1}]        evaluation vectors, each x_j ∈ Z_q^r
//   f   ∈ Z_q^{r^ℓ · κ · n}      committed polynomial (τ = 1 throughout)
//   t   ∈ Z_q^{κ · n}             commitment
//   u   ∈ Z_q^{κ · n}             evaluation claim
//   s_j ∈ Z^{r^{j+1} · κ · n · α}  opening vectors, j = 0…ℓ−1
//
//   Challenge  C_{i+1} ∈ {0,1}^{r·κ × κ}
// ============================================================

#[derive(Clone)]
struct Params {
    q:     i64,      // modulus
    n:     usize,    // SIS matrix height
    r:     usize,    // folding factor
    kappa: usize,    // statistical security parameter
    alpha: usize,    // ⌈log_δ q⌉
    delta: i64,      // gadget base (typically 2)
    ell:   usize,    // tree depth = number of protocol rounds
    // betas[i] = (rκ)^i · (δ−1): ℓ∞ norm bound for y_i in round i.
    // After i binary-matrix folds each of width rκ, the bound grows by that factor.
    betas: Vec<i64>,
    a:     Mat,      // n × (r·n·α)
    // x[j] ∈ Z_q^r: evaluation row-vector for tree level j (j = 0 innermost).
    // The committed polynomial evaluates to u at the "point" encoded by x[0..ℓ-1].
    x:     Vec<Vec1>,
}

impl Params {
    fn f_dim(&self) -> usize { self.r.pow(self.ell as u32) * self.kappa * self.n }
    fn kn(&self)    -> usize { self.kappa * self.n }
}

// ============================================================
// 5.  Evaluation witness  (used by the prover each round)
//
//   eval_witness(f^{(i)}, &x[0..depth], q, r)  →  v_i ∈ Z_q^{r·κ·n}
//
//   where depth = ℓ − i − 1.
//
//   The function reduces f by applying x[0] innermost first:
//     step 0: (I_{|f|/r} ⊗ x[0]ᵀ) · f  — contracts the innermost r-block per entry
//     step 1: same with x[1], etc.
//   After `depth` steps the result has length r·κ·n.
//
//   At commit time v_0 is returned, and u = (I_{κn} ⊗ x[ℓ-1]) · v_0.
//   On round i the prover re-derives v_i from the folded f^{(i)}.
// ============================================================

/// Reduce f (length r^{depth+1}·base) to length r·base by applying each x_j
/// (innermost first) as a 1×r row vector.
fn eval_witness(f: &Vec1, xs: &[Vec1], q: i64, r: usize) -> Vec1 {
    if xs.is_empty() {
        return f.clone();
    }
    // f.len() must be an exact multiple of r.
    debug_assert_eq!(f.len() % r, 0, "eval_witness: f.len() must be divisible by r");
    let n_out  = f.len() / r;
    // (I_{n_out} ⊗ x_row) where x_row is a 1×r matrix
    let x_row  = vec![xs[0].clone()];           // 1 × r
    let kmat   = kron(&eye(n_out), &x_row);      // n_out × (n_out·r)
    let f_next = mv(&kmat, f, q);               // length n_out
    eval_witness(&f_next, &xs[1..], q, r)
}

// ============================================================
// 6.  Commitment  (Section 4, eq. 2)
//
// Tree built bottom-up:
//   s_{ℓ−1} = G^{−1}_{r^ℓ κ n}(f)
//   s_j     = G^{−1}_{r^{j+1} κ n}( (I_{r^{j+1} κ} ⊗ A) · s_{j+1} )  j = ℓ−2…0
//   t       = (I_κ ⊗ A) · s_0
//   u       = (I_{κn} ⊗ x[ℓ-1]) · eval_witness(f, x[0..ℓ-2])
// ============================================================

#[derive(Clone)]
struct CommitOutput {
    t: Vec1,       // ∈ Z_q^{κn}    — SIS commitment
    u: Vec1,       // ∈ Z_q^{κn}    — evaluation claim
    s: Vec<Vec1>,  // s[j] ∈ Z^{r^{j+1}·κ·n·α}
    f: Vec1,       // ∈ Z_q^{r^ℓ·κ·n}
}

fn commit(p: &Params, f: &Vec1) -> CommitOutput {
    assert_eq!(f.len(), p.f_dim(), "commit: f must have length {}", p.f_dim());

    // Build s-array bottom-up, then reverse so s[0] = s_0.
    let mut s_rev: Vec<Vec1> = Vec::with_capacity(p.ell);

    // s_{ℓ−1} = G^{−1}(f)
    s_rev.push(g_inv(f, p.q, p.delta, p.alpha));

    for step in 1..p.ell {
        let jp1   = p.ell - step;                     // j+1 in the recurrence
        let i_dim = p.r.pow(jp1 as u32) * p.kappa;    // size of I block
        let kia   = kron(&eye(i_dim), &p.a);           // (i_dim·n) × (i_dim·r·n·α)
        let prev  = s_rev.last().unwrap();
        let tmp   = mv(&kia, prev, p.q);
        s_rev.push(g_inv(&tmp, p.q, p.delta, p.alpha));
    }
    s_rev.reverse();                                   // s[0] = s_0 … s[ℓ-1] = s_{ℓ-1}

    // t = (I_κ ⊗ A) · s_0
    let ktop = kron(&eye(p.kappa), &p.a);
    let t    = mv(&ktop, &s_rev[0], p.q);

    // v_0 = eval_witness(f, x[0..ℓ-2])  reduces f to length rκn
    let v0   = eval_witness(f, &p.x[..p.ell - 1], p.q, p.r);
    // u = (I_{κn} ⊗ x[ℓ-1]) · v_0
    let x_top  = vec![p.x[p.ell - 1].clone()];        // 1 × r
    let ku_mat = kron(&eye(p.kn()), &x_top);           // κn × rκn
    let u      = mv(&ku_mat, &v0, p.q);

    CommitOutput { t, u, s: s_rev, f: f.clone() }
}

/// Verify a commitment opening.
fn verify_commit(p: &Params, c: &CommitOutput) -> bool {
    // 1. t = (I_κ ⊗ A) · s_0
    if mv(&kron(&eye(p.kappa), &p.a), &c.s[0], p.q) != c.t {
        eprintln!("verify_commit: t mismatch");
        return false;
    }
    // 2. G_{r^{j+1}·κ·n} · s_j  =  (I_{r^{j+1}·κ} ⊗ A) · s_{j+1}  for j = 0..ℓ-2
    for j in 0..p.ell - 1 {
        let m_g  = p.r.pow((j + 1) as u32) * p.kappa * p.n;
        let lhs  = mv(&gadget(m_g, p.alpha, p.delta), &c.s[j], p.q);
        let i_dim = p.r.pow((j + 1) as u32) * p.kappa;
        let rhs  = mv(&kron(&eye(i_dim), &p.a), &c.s[j + 1], p.q);
        if lhs != rhs {
            eprintln!("verify_commit: recurrence mismatch at j={}", j);
            return false;
        }
    }
    // 3. G_{r^ℓ·κ·n} · s_{ℓ-1}  =  f
    let m_fin = p.r.pow(p.ell as u32) * p.kappa * p.n;
    if mv(&gadget(m_fin, p.alpha, p.delta), &c.s[p.ell - 1], p.q) != c.f {
        eprintln!("verify_commit: G·s_{{ℓ-1}} ≠ f");
        return false;
    }
    // 4. norm bounds: G^{-1} outputs are in {0,…,δ-1}
    for (j, sj) in c.s.iter().enumerate() {
        if linf(sj) >= p.delta {
            eprintln!("verify_commit: norm exceeded at j={}", j);
            return false;
        }
    }
    true
}

// ============================================================
// 7.  Folding protocol  (Section 5, Figure 6)
//
// Statement  x^{(i)} = (t^{(i)}, u^{(i)}) ∈ Z_q^{κn} × Z_q^{κn}
// Witness    w^{(i)} = (s^{(i)}_0, …, s^{(i)}_{ℓ-i-1}, f^{(i)})
//
// Each round i:
//   Prover sends  y_i = s^{(i)}_0         (SIS opening)
//                 v_i = eval_witness(f^{(i)}, x[0..ℓ-i-2]) ∈ Z_q^{rκn}
//
//   Verifier checks:
//     b₀: (I_κ ⊗ A) · y_i  =  t^{(i)}
//     b₁: ‖y_i‖_∞ ≤ β_i
//     b₂: (I_{κn} ⊗ x[ℓ-1-i]) · v_i  =  u^{(i)}
//
//   Challenge  C_{i+1} ∈ {0,1}^{rκ × κ}
//
//   Next statement:
//     t^{(i+1)} = (C^T ⊗ I_n) · G_{rκn} · y_i
//     u^{(i+1)} = (C^T ⊗ I_n) · v_i
//
//   Next witness:
//     s^{(i+1)}_j = (C^T ⊗ I_{r^{j+1}·n·α}) · s^{(i)}_{j+1}  j=0…(remaining-2)
//     f^{(i+1)}   = (C^T ⊗ I_{r^{depth-1}·n}) · f^{(i)}
// ============================================================

#[derive(Clone)]
struct Statement {
    t: Vec1,   // ∈ Z_q^{κn}
    u: Vec1,   // ∈ Z_q^{κn}
}

#[derive(Clone)]
struct Witness {
    s: Vec<Vec1>, // s[0] sent this round; drops by one each fold
    f: Vec1,      // current folded message
}

/// Uniform binary matrix in {0,1}^{rκ × κ}.
fn sample_challenge(p: &Params) -> Mat {
    let mut rng = thread_rng();
    (0..p.r * p.kappa)
        .map(|_| (0..p.kappa).map(|_| rng.gen_range(0i64..2)).collect())
        .collect()
}

/// All three verifier checks for round `i`.
fn verifier_ok(p: &Params, stmt: &Statement, y: &Vec1, v: &Vec1, round: usize) -> bool {
    // b₀: (I_κ ⊗ A) · y = t
    if mv(&kron(&eye(p.kappa), &p.a), y, p.q) != stmt.t {
        eprintln!("b₀ failed in round {}", round);
        return false;
    }
    // b₁: ‖y‖_∞ ≤ β_i
    if linf(y) > p.betas[round] {
        eprintln!("b₁ failed in round {} — ‖y‖_∞={} > β={}", round, linf(y), p.betas[round]);
        return false;
    }
    // b₂: (I_{κn} ⊗ x[ℓ-1-i]) · v = u
    let xi    = vec![p.x[p.ell - 1 - round].clone()]; // 1 × r
    let kx    = kron(&eye(p.kn()), &xi);               // κn × rκn
    if mv(&kx, v, p.q) != stmt.u {
        eprintln!("b₂ failed in round {} — eval mismatch", round);
        return false;
    }
    true
}

/// Compute next statement from y_i, v_i, and challenge C.
///   t^{(i+1)} = (C^T ⊗ I_n) · G_{rκn} · y_i
///   u^{(i+1)} = (C^T ⊗ I_n) · v_i
fn next_statement(p: &Params, y: &Vec1, v: &Vec1, c: &Mat) -> Statement {
    let ct   = transpose(c);                                         // κ × rκ
    let kct  = kron(&ct, &eye(p.n));                                 // κn × rκn
    let m    = p.r * p.kappa * p.n;
    let gy   = mv(&gadget(m, p.alpha, p.delta), y, p.q);            // rκn
    Statement {
        t: mv(&kct, &gy, p.q),
        u: mv(&kct, v,  p.q),
    }
}

/// Fold the witness.  `depth` = rounds still remaining (including this one).
fn next_witness(p: &Params, w: &Witness, c: &Mat, depth: usize) -> Witness {
    let ct = transpose(c); // κ × rκ

    // s^{(i+1)}_j = (C^T ⊗ I_{r^{j+1}·n·α}) · s^{(i)}_{j+1}
    let new_s: Vec<Vec1> = (0..w.s.len() - 1).map(|j| {
        let blk = p.r.pow((j + 1) as u32) * p.n * p.alpha;
        mv(&kron(&ct, &eye(blk)), &w.s[j + 1], 0) // integer arithmetic
    }).collect();

    // f^{(i+1)} = (C^T ⊗ I_{r^{depth-1}·n}) · f^{(i)}
    let fblk  = p.r.pow((depth - 1) as u32) * p.n;
    let new_f = mv(&kron(&ct, &eye(fblk)), &w.f, p.q);

    Witness { s: new_s, f: new_f }
}

/// Run all ℓ rounds.  Returns true iff every round passes all three checks.
fn run_protocol(p: &Params, comm: &CommitOutput) -> bool {
    let mut stmt = Statement { t: comm.t.clone(), u: comm.u.clone() };
    let mut wit  = Witness   { s: comm.s.clone(), f: comm.f.clone() };

    for i in 0..p.ell {
        let depth = p.ell - i;

        // Prover computes both messages.
        let y = wit.s[0].clone();
        // eval_witness reduces f^{(i)} (length rκn · r^{depth-1}) to rκn.
        // Inner levels consumed: depth-1 evaluation vectors x[0..depth-2].
        let v = eval_witness(&wit.f, &p.x[..depth - 1], p.q, p.r);

        if !verifier_ok(p, &stmt, &y, &v, i) {
            return false;
        }

        if i < p.ell - 1 {
            let c = sample_challenge(p);
            stmt  = next_statement(p, &y, &v, &c);
            wit   = next_witness(p, &wit, &c, depth);
        }
    }
    true
}

// ============================================================
// 8.  Helpers
// ============================================================

fn rand_vec(n: usize, q: i64) -> Vec1 {
    let mut rng = thread_rng();
    (0..n).map(|_| rng.gen_range(0..q)).collect()
}

fn make_params(ell: usize) -> Params {
    // Toy parameters: q=8, δ=2, α=3 (2³=8=q), n=2, r=2, κ=1.
    let q = 8i64; let delta = 2i64; let alpha = 3;
    let n = 2; let r = 2; let kappa = 1;
    // Fixed A for reproducibility.
    let a_cols = r * n * alpha; // 12
    let a: Mat = (0..n)
        .map(|i| (0..a_cols).map(|j| ((i + j) as i64) % q).collect())
        .collect();
    // β_i = (rκ)^i · (δ−1)
    let betas: Vec<i64> = (0..ell)
        .map(|i| (r as i64 * kappa as i64).pow(i as u32) * (delta - 1))
        .collect();
    // Evaluation vectors x[j] ∈ Z_q^r.
    // Using x[j] = [1, z^{2^j}] with z = 3 gives a concrete multilinear evaluation point.
    let z = 3i64;
    let x: Vec<Vec1> = (0..ell)
        .map(|j| {
            let zj = modq(z.pow(j as u32), q);
            (0..r).map(|k| modq(zj.pow(k as u32), q)).collect()
        })
        .collect();
    Params { q, n, r, kappa, alpha, delta, ell, betas, a, x }
}

// ============================================================
// 9.  Main — demo
// ============================================================

fn main() {
    println!("=== Lattice Polynomial Commitment Demo ===");

    let p = make_params(2);
    let f = rand_vec(p.f_dim(), p.q);

    println!("n={} r={} κ={} α={} δ={} ℓ={} q={}", p.n, p.r, p.kappa, p.alpha, p.delta, p.ell, p.q);
    println!("eval points x = {:?}", p.x);
    println!("f (len={})  = {:?}", f.len(), f);

    let comm = commit(&p, &f);
    println!("t = {:?}", comm.t);
    println!("u = {:?}", comm.u);

    let ok_commit = verify_commit(&p, &comm);
    println!("verify_commit → {}", ok_commit);

    let ok_proto = run_protocol(&p, &comm);
    println!("run_protocol  → {}", ok_proto);

    if ok_commit && ok_proto { println!("All checks passed."); }
    else { eprintln!("FAILURE."); }
}

// ============================================================
// 10.  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── gadget primitives ──────────────────────────────────────

    #[test]
    fn gadget_structure_2_3_2() {
        let g = gadget(2, 3, 2);
        assert_eq!(g[0], vec![1, 2, 4, 0, 0, 0]);
        assert_eq!(g[1], vec![0, 0, 0, 1, 2, 4]);
    }

    #[test]
    fn g_inv_roundtrip() {
        let p = make_params(1);
        for &x in &[0i64, 1, 3, 5, 7] {
            let d    = g_inv(&vec![x], p.q, p.delta, p.alpha);
            let back = mv(&gadget(1, p.alpha, p.delta), &d, p.q);
            assert_eq!(back[0], modq(x, p.q), "roundtrip failed for x={}", x);
        }
    }

    #[test]
    fn g_inv_multi() {
        let p = make_params(1);
        let v    = vec![0i64, 3, 6, 7];
        let d    = g_inv(&v, p.q, p.delta, p.alpha);
        let back = mv(&gadget(v.len(), p.alpha, p.delta), &d, p.q);
        assert_eq!(back, v);
    }

    #[test]
    fn g_inv_digits_in_range() {
        let p = make_params(1);
        for digit in g_inv(&rand_vec(8, p.q), p.q, p.delta, p.alpha) {
            assert!(digit >= 0 && digit < p.delta);
        }
    }

    // ── eval_witness ──────────────────────────────────────────

    #[test]
    fn eval_witness_ell1_trivial() {
        // depth = 0: eval_witness(f, &[], ...) = f
        let p = make_params(1);
        let f = rand_vec(p.r * p.kn(), p.q);
        let v = eval_witness(&f, &[], p.q, p.r);
        assert_eq!(v, f);
    }

    #[test]
    fn eval_witness_ell2_dimension() {
        // depth = 1: f ∈ Z_q^{r²κn}, v ∈ Z_q^{rκn}
        let p  = make_params(2);
        let f  = rand_vec(p.f_dim(), p.q);           // r²κn = 8
        let v0 = eval_witness(&f, &p.x[..1], p.q, p.r); // consume x[0]
        assert_eq!(v0.len(), p.r * p.kn());           // rκn = 4
    }

    #[test]
    fn eval_witness_linearity() {
        // eval_witness is linear in f: eval(a·f + b·g) = a·eval(f) + b·eval(g)  (mod q)
        let p  = make_params(2);
        let q  = p.q;
        let f1 = rand_vec(p.f_dim(), q);
        let f2 = rand_vec(p.f_dim(), q);
        let sum: Vec1 = f1.iter().zip(&f2).map(|(&a, &b)| modq(a + b, q)).collect();

        let v1  = eval_witness(&f1,  &p.x[..1], q, p.r);
        let v2  = eval_witness(&f2,  &p.x[..1], q, p.r);
        let vss = eval_witness(&sum, &p.x[..1], q, p.r);
        let expected: Vec1 = v1.iter().zip(&v2).map(|(&a, &b)| modq(a + b, q)).collect();
        assert_eq!(vss, expected, "eval_witness is not linear");
    }

    #[test]
    fn b2_holds_at_commit() {
        // Check that (I_{κn} ⊗ x[ℓ-1]) · v₀ = u^{(0)} as computed by commit.
        let p   = make_params(2);
        let f   = rand_vec(p.f_dim(), p.q);
        let c   = commit(&p, &f);
        let v0  = eval_witness(&f, &p.x[..p.ell - 1], p.q, p.r);
        let xi  = vec![p.x[p.ell - 1].clone()];
        let kx  = kron(&eye(p.kn()), &xi);
        let u_check = mv(&kx, &v0, p.q);
        assert_eq!(u_check, c.u, "b₂ pre-check: u mismatch at commit time");
    }

    #[test]
    fn b2_folds_correctly() {
        // After one fold: (I_{κn} ⊗ x[ℓ-2]) · v₁ == u^{(1)} = (C^T ⊗ I_n) · v₀
        // This verifies the algebraic identity the protocol relies on.
        let p    = make_params(2);
        let q    = p.q;
        let f    = rand_vec(p.f_dim(), q);
        let v0   = eval_witness(&f, &p.x[..1], q, p.r); // depth 1

        // Sample a challenge and fold.
        let c_mat = sample_challenge(&p);
        let ct    = transpose(&c_mat);
        // u^{(1)} = (C^T ⊗ I_n) · v₀
        let u1    = mv(&kron(&ct, &eye(p.n)), &v0, q);
        // f^{(1)} = (C^T ⊗ I_{rn}) · f
        let f1    = mv(&kron(&ct, &eye(p.r * p.n)), &f, q);
        // v₁ = eval_witness(f^{(1)}, [], ..) = f^{(1)}  (depth 0)
        let v1    = eval_witness(&f1, &[], q, p.r);
        // Check: (I_{κn} ⊗ x[0]) · v₁ = u^{(1)}
        let xi    = vec![p.x[0].clone()];
        let kx    = kron(&eye(p.kn()), &xi);
        assert_eq!(mv(&kx, &v1, q), u1, "b₂ fold invariant failed");
    }

    // ── commitment ────────────────────────────────────────────

    #[test]
    fn commit_verify_ell1() {
        let p = make_params(1);
        assert!(verify_commit(&p, &commit(&p, &rand_vec(p.f_dim(), p.q))));
    }

    #[test]
    fn commit_verify_ell2() {
        let p = make_params(2);
        assert!(verify_commit(&p, &commit(&p, &rand_vec(p.f_dim(), p.q))));
    }

    #[test]
    fn commit_verify_ell3() {
        let p = make_params(3);
        assert!(verify_commit(&p, &commit(&p, &rand_vec(p.f_dim(), p.q))));
    }

    #[test]
    fn tampered_t_rejected() {
        let p = make_params(2);
        let mut c = commit(&p, &rand_vec(p.f_dim(), p.q));
        c.t[0] = (c.t[0] + 1) % p.q;
        assert!(!verify_commit(&p, &c));
    }

    #[test]
    fn tampered_f_rejected() {
        let p = make_params(2);
        let mut c = commit(&p, &rand_vec(p.f_dim(), p.q));
        c.f[0] = (c.f[0] + 1) % p.q;
        assert!(!verify_commit(&p, &c));
    }

    #[test]
    fn tampered_s0_rejected() {
        let p = make_params(2);
        let mut c = commit(&p, &rand_vec(p.f_dim(), p.q));
        c.s[0][0] = (c.s[0][0] + 1) % p.q;
        assert!(!verify_commit(&p, &c));
    }

    // ── full protocol ─────────────────────────────────────────

    #[test]
    fn protocol_honest_ell1() {
        let p = make_params(1);
        let c = commit(&p, &rand_vec(p.f_dim(), p.q));
        assert!(run_protocol(&p, &c));
    }

    #[test]
    fn protocol_honest_ell2() {
        let p = make_params(2);
        let c = commit(&p, &rand_vec(p.f_dim(), p.q));
        assert!(run_protocol(&p, &c));
    }

    #[test]
    fn protocol_honest_ell3() {
        let p = make_params(3);
        let c = commit(&p, &rand_vec(p.f_dim(), p.q));
        assert!(run_protocol(&p, &c));
    }

    #[test]
    fn protocol_tampered_t_fails() {
        let p = make_params(2);
        let mut c = commit(&p, &rand_vec(p.f_dim(), p.q));
        c.t[0] = (c.t[0] + 1) % p.q;
        assert!(!run_protocol(&p, &c));
    }

    #[test]
    fn protocol_tampered_u_fails() {
        let p = make_params(2);
        let mut c = commit(&p, &rand_vec(p.f_dim(), p.q));
        c.u[0] = (c.u[0] + 1) % p.q; // wrong evaluation claim
        assert!(!run_protocol(&p, &c));
    }

    // ── Kronecker sanity ──────────────────────────────────────

    #[test]
    fn kron_mixed_product() {
        let a: Mat = vec![vec![1, 2], vec![3, 4]];
        let b: Mat = vec![vec![0, 1], vec![1, 0]];
        let c: Mat = vec![vec![1, 0], vec![0, 1]]; // identity
        let d: Mat = vec![vec![2, 0], vec![0, 2]]; // 2·I

        let ab  = kron(&a, &b);
        let cd  = kron(&c, &d);
        let lhs = {
            let (r, cc, k) = (rows(&ab), cols(&cd), cols(&ab));
            let mut out = mat_zero(r, cc);
            for i in 0..r { for j in 0..cc { for l in 0..k {
                out[i][j] += ab[i][l] * cd[l][j];
            }}}
            out
        };
        // (A·I) ⊗ (B·2I) = A ⊗ 2B
        let rhs = kron(&a, &vec![vec![0, 2], vec![2, 0]]);
        assert_eq!(lhs, rhs);
    }
}
