use rand::Rng;
use num_bigint::BigInt;
use num_traits::{Zero, One};
use num_traits::Signed;

// ========================================================
// 1) Basic mod q, vectors, matrices
// ========================================================
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModQ {
    pub value: BigInt,
    pub q: BigInt,
}

impl ModQ {
    pub fn new(value: BigInt, q: &BigInt) -> ModQ {
        let mut val = value % q;
        if val < BigInt::zero() {
            val += q;
        }
        ModQ { value: val, q: q.clone() }
    }

    pub fn zero(q: &BigInt) -> Self {
        Self::new(BigInt::zero(), q)
    }
    pub fn add_mod(&self, other: &ModQ) -> ModQ {
        let mut s = &self.value + &other.value;
        s %= &self.q;
        if s < BigInt::zero() {
            s += &self.q;
        }
        ModQ::new(s, &self.q)
    }
    pub fn mul_mod(&self, other: &ModQ) -> ModQ {
        let mut prod = &self.value * &other.value;
        prod %= &self.q;
        if prod < BigInt::zero() {
            prod += &self.q;
        }
        ModQ::new(prod, &self.q)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecModQ {
    pub data: Vec<ModQ>,
    pub q: BigInt,
}

impl VecModQ {
    pub fn new(data: Vec<ModQ>, q: &BigInt) -> Self {
        for d in &data {
            assert_eq!(d.q, *q);
        }
        VecModQ { data, q: q.clone() }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn zero(len: usize, q: &BigInt) -> Self {
        let mut arr = Vec::with_capacity(len);
        for _ in 0..len {
            arr.push(ModQ::zero(q));
        }
        VecModQ::new(arr, q)
    }
    pub fn norm(&self) -> BigInt {
        self.data
            .iter()
            .map(|v| v.value.abs()) // Take the absolute value of each element
            .max()                  // Find the maximum value
            .unwrap_or(BigInt::zero()) // Return 0 if the vector is empty
    }
}

#[derive(Clone, Debug)]
pub struct MatModQ {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<ModQ>,
    pub q: BigInt,
}

impl MatModQ {
    pub fn identity(size: usize, q: &BigInt) -> MatModQ {
        let mut data = Vec::new();
        for i in 0..size {
            for j in 0..size {
                if i == j {
                    data.push(ModQ::new(BigInt::one(), q));
                } else {
                    data.push(ModQ::zero(q));
                }
            }
        }
        MatModQ::new(size, size, data, q)
    }
    pub fn mul_vec(&self, v: &VecModQ) -> VecModQ {
        assert_eq!(
            self.cols,
            v.len(),
            "Matrix-vector multiplication dimension mismatch: matrix.cols = {}, vector.len = {}",
            self.cols,
            v.len()
        );

        let mut out = VecModQ::zero(self.rows, &self.q);
        for r in 0..self.rows {
            let mut acc = ModQ::zero(&self.q);
            for c in 0..self.cols {
                let idx = r * self.cols + c;
                assert!(
                    idx < self.data.len(),
                    "Index out of bounds: idx = {}, matrix data.len = {}",
                    idx,
                    self.data.len()
                );
                let prod = self.data[idx].mul_mod(&v.data[c]);
                acc = acc.add_mod(&prod);
            }
            out.data[r] = acc;
        }
        out
    }
    pub fn new(rows: usize, cols: usize, data: Vec<ModQ>, q: &BigInt) -> MatModQ {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length must match rows * cols"
        );
        MatModQ {
            rows,
            cols,
            data,
            q: q.clone(),
        }
    }
    pub fn tensor_product(&self, other: &MatModQ) -> MatModQ {
        let new_rows = self.rows * other.rows;
        let new_cols = self.cols * other.cols;
        let mut data = Vec::with_capacity(new_rows * new_cols);

        for r1 in 0..self.rows {
            for r2 in 0..other.rows {
                for c1 in 0..self.cols {
                    for c2 in 0..other.cols {
                        let idx1 = r1 * self.cols + c1;
                        let idx2 = r2 * other.cols + c2;
                        let value = self.data[idx1].mul_mod(&other.data[idx2]);
                        data.push(value);
                    }
                }
            }
        }

        MatModQ::new(new_rows, new_cols, data, &self.q)
    }
}



//
// ========================================================
// 2) Gadget & Inverse (powers-of-two diagonal)
// ========================================================
fn gadget_matrix(m: usize, q: &BigInt) -> MatModQ {
    // diag(2^0, 2^1, ..., 2^(m-1))
    let mut data = Vec::with_capacity(m * m);

    for r in 0..m {
        for c in 0..m {
            if r == c {
                // Place powers of 2 on the diagonal
                let val = BigInt::from(1) << r; // 2^r
                data.push(ModQ::new(val, q));
            } else {
                // Off-diagonal elements are zero
                data.push(ModQ::zero(q));
            }
        }
    }

    MatModQ::new(m, m, data, q)
}


fn g_inv_apply(v: &VecModQ) -> VecModQ {
    // inverse for powers-of-two diagonal
    let q = v.q.clone();
    let mut out = Vec::new();
    for (i, coord) in v.data.iter().enumerate() {
        let pow2i = BigInt::from(1) << i;
        if let Some(inv_2i) = mod_inverse(&pow2i, &q) {
            let new_val = (&coord.value * &inv_2i) % &q;
            out.push(ModQ::new(new_val, &q));
        } else {
            // fallback
            out.push(ModQ::zero(&q));
        }
    }
    VecModQ::new(out, &q)
}

fn construct_ct(challenge: &Challenge, dim: usize, q: &BigInt) -> MatModQ {
    assert!(
        challenge.data.len() <= dim,
        "Challenge dimension mismatch: challenge.len = {}, dim = {}",
        challenge.data.len(),
        dim
    );

    let mut data = Vec::new();
    for i in 0..dim {
        let val = if i < challenge.data.len() {
            challenge.data.data[i].value.clone() // Corrected access
        } else {
            BigInt::zero() // Fill extra dimensions with zeros
        };
        data.push(ModQ::new(val, q));
    }

    MatModQ::new(dim, dim, data, q)
}


fn generate_xj(dim: usize, q: &BigInt) -> MatModQ {
    // Creates a diagonal matrix with entries 2^0, 2^1, ..., 2^(dim-1)
    let mut data = Vec::new();
    for i in 0..dim {
        for j in 0..dim {
            if i == j {
                let val = BigInt::from(1) << i; // 2^i
                data.push(ModQ::new(val, q));
            } else {
                data.push(ModQ::zero(q));
            }
        }
    }
    MatModQ::new(dim, dim, data, q)
}




fn mod_inverse(a: &BigInt, m: &BigInt) -> Option<BigInt> {
    if m.is_zero() { return None; }
    let (g, x, _) = extended_gcd(a.clone(), m.clone());
    if g == BigInt::one() {
        let mut r = x % m;
        if r < BigInt::zero() {
            r += m;
        }
        Some(r)
    } else {
        None
    }
}

fn extended_gcd(a: BigInt, b: BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        return (a.clone(), BigInt::one(), BigInt::zero());
    }
    let (g, x1, y1) = extended_gcd(b.clone(), &a % &b);
    let q = &a / &b;
    let x = y1.clone();
    let y = x1 - &q * &y1;
    (g, x, y)
}

//
// ========================================================
// 3) commit_t with Reverse expansions
//    We'll do expansions in reverse order from r_{ell+1} down to r_1
// ========================================================

#[derive(Clone)]
pub struct CommitmentParams {
    pub q: BigInt,
    pub A: MatModQ,
    pub r_vals: Vec<usize>, // r_1..r_{ell+1}
    pub ell: usize,
    pub beta: i64,
    pub kappa: usize, // Add this
    pub tau: usize,   // Add this
    pub n: usize,     // Add this
}

fn compute_t(params: &CommitmentParams, f: &VecModQ) -> VecModQ {
    let mut v = f.clone();
    for &r_i in params.r_vals.iter().rev() {
        // G_{r_i}^{-1}
        v = g_inv_apply(&v);

        // (I_{r_i} ⊗ A)
        let dim = r_i;
        let matA = &params.A;
        let expected_in = dim * matA.cols;
        assert_eq!(
            v.len(),
            expected_in,
            "compute_t(reverse): mismatch in (I_{r_i} ⊗ A)*v. expected {}, got {}",
            expected_in,
            v.len()
        );
        // output => dim*matA.rows
        let mut out = VecModQ::zero(dim*matA.rows, &v.q);
        for d in 0..dim {
            for rr in 0..matA.rows {
                let mut acc = ModQ::zero(&v.q);
                for cc in 0..matA.cols {
                    let idx_v = d*matA.cols + cc;
                    let idx_a = rr*matA.cols + cc;
                    let prod = matA.data[idx_a].mul_mod(&v.data[idx_v]);
                    acc = acc.add_mod(&prod);
                }
                out.data[d*matA.rows + rr] = acc;
            }
        }
        v = out;
    }
    v
}

fn compute_s_j(params: &CommitmentParams, f: &VecModQ, j: usize) -> VecModQ {
    let mut v = f.clone();
    let start = params.r_vals.len(); // = ell+1
    let stop  = j+1;                 // inclusive

    for i in (stop..=start).rev() {
        let idx = i - 1;
        let r_i = params.r_vals[idx];

        // 1) G_{r_i}^{-1}
        v = g_inv_apply(&v);

        // 2) (I_{r_i} \otimes A)
        let dim = r_i;
        let matA = &params.A;
        let expected_len = dim * matA.cols;
        assert_eq!(
            v.len(),
            expected_len,
            "compute_s_j(reverse): dimension mismatch at i={}, r_i={}. Expected {}, got {}",
            i,
            r_i,
            expected_len,
            v.len()
        );

        let mut out = VecModQ::zero(dim * matA.rows, &v.q);
        for d in 0..dim {
            for rr in 0..matA.rows {
                let mut acc = ModQ::zero(&v.q);
                for cc in 0..matA.cols {
                    let idx_v = d*matA.cols + cc;
                    let idx_a = rr*matA.cols + cc;
                    let prod = matA.data[idx_a].mul_mod(&v.data[idx_v]);
                    acc = acc.add_mod(&prod);
                }
                out.data[d * matA.rows + rr] = acc;
            }
        }
        v = out;
    }
    v
}

#[derive(Clone)]
pub struct CommitOutput {
    pub t: VecModQ,
    pub s_arr: Vec<VecModQ>,
    pub f: VecModQ,
}

fn commit_section4(params: &CommitmentParams, f: &VecModQ) -> CommitOutput {
    let t = compute_t(params, f);

    let mut s_arr = Vec::new();
    for j in 0..params.ell {
        let sj = compute_s_j(params, f, j);
        s_arr.push(sj);
    }
    CommitOutput { t, s_arr, f: f.clone() }
}

fn open_section4(params: &CommitmentParams, comm: &CommitOutput) -> bool {
    let t_check = compute_t(params, &comm.f);
    if t_check != comm.t {
        eprintln!("open_section4: mismatch t");
        return false;
    }
    for j in 0..comm.s_arr.len() {
        let sj_check = compute_s_j(params, &comm.f, j);
        if comm.s_arr[j] != sj_check {
            eprintln!("open_section4: mismatch s_j, j={}", j);
            return false;
        }
    }
    true
}

//
// ========================================================
// 4) (2ℓ+1)-msg argument, Section 5 (inline checks)
// ========================================================
#[derive(Clone)]
pub struct Instance {
    pub A: MatModQ,
    pub t_i: VecModQ,
    pub u_i: VecModQ, // Add this
}

#[derive(Clone)]
pub struct Witness {
    pub s_j: Vec<VecModQ>,
    pub f: VecModQ,
}

/// Minimal challenge structure.
#[derive(Clone)]
pub struct Challenge {
    pub data: VecModQ,
}

fn prover_round_i(pp: &CommitmentParams, x_i: &Instance, w_i: &Witness) -> VecModQ {
    // Compute y_i = s₀^(i)
    let y_i = w_i.s_j[0].clone();

    let mut v_i = w_i.f.clone();
    for j in 0..pp.ell {
        let I_rj_kn = MatModQ::identity(pp.r_vals[j] * pp.kappa * pp.n, &pp.q);
        let X_l_i_j = generate_xj(pp.r_vals[j], &pp.q);
        let tensor = I_rj_kn.tensor_product(&X_l_i_j);
        v_i = tensor.mul_vec(&v_i);
}


    // Combine y_i and v_i
    let mut z_i_data = y_i.data.clone();
    z_i_data.extend(v_i.data);
    VecModQ::new(z_i_data, &pp.q)
}


fn verifier_round_i(pp: &CommitmentParams, x_i: &Instance, z_i: &VecModQ) -> bool {
    // Extract y_i and v_i from z_i
    let y_i_len = x_i.A.rows;
    let y_i = VecModQ::new(z_i.data[0..y_i_len].to_vec(), &pp.q);
    let v_i = VecModQ::new(z_i.data[y_i_len..].to_vec(), &pp.q);

    assert_eq!(
        z_i.len(),
        y_i_len + v_i.len(),
        "verifier_round_i: z_i length mismatch. Expected {}, got {}",
        y_i_len + v_i.len(),
        z_i.len()
    );

    // Check b_i,0: [I_κτ ⊗ A] ⋅ y_i == t^(i)
    let I_kappa_tau = MatModQ::identity(pp.kappa * pp.tau, &pp.q);
    let tensor_y = I_kappa_tau.tensor_product(&x_i.A);
    assert_eq!(
        tensor_y.rows,
        x_i.t_i.len(),
        "verifier_round_i: b_i,0 tensor dimension mismatch"
    );
    if tensor_y.mul_vec(&y_i) != x_i.t_i {
        eprintln!(
            "verifier_round_i: b_i,0 failed. Expected t^(i) = {:?}, but got {:?}",
            x_i.t_i,
            tensor_y.mul_vec(&y_i)
        );
        return false;
    }

    // Check b_i,1: [I_kn ⊗ X_{ℓ−i}] ⋅ v_i == u^(i)
    let I_kn = MatModQ::identity(pp.kappa * pp.n, &pp.q);
    let X_l_i = generate_xj(pp.kappa, &pp.q);
    let tensor_v = I_kn.tensor_product(&X_l_i);
    assert_eq!(
        tensor_v.cols,
        v_i.len(),
        "verifier_round_i: b_i,1 tensor dimension mismatch"
    );
    if tensor_v.mul_vec(&v_i) != x_i.u_i {
        eprintln!(
            "verifier_round_i: b_i,1 failed. Expected u^(i) = {:?}, but got {:?}",
            x_i.u_i,
            tensor_v.mul_vec(&v_i)
        );
        return false;
    }

    // Check b_i,2: ||y_i|| ≤ β_{ℓ−i}
    if y_i.norm() > BigInt::from(pp.beta) {
        eprintln!(
            "verifier_round_i: b_i,2 failed (||y_i|| = {}, β = {})",
            y_i.norm(),
            pp.beta
        );
        return false;
    }

    // All checks passed
    true
}


fn next_instance(x_i: &Instance, z_i: &VecModQ, c: &Challenge) -> Instance {
    let C_t = construct_ct(c, x_i.t_i.len(), &x_i.A.q);

    // Update t^{(i+1)}: C_t ⊗ I ⋅ G ⋅ z_i
    let G_rk_kn = gadget_matrix(x_i.A.cols, &x_i.A.q);
    let updated_t = C_t.tensor_product(&MatModQ::identity(x_i.t_i.len(), &x_i.A.q))
                      .mul_vec(&G_rk_kn.mul_vec(z_i));

    // Update u^{(i+1)}: C_t ⊗ I ⋅ z_i
    let updated_u = C_t.tensor_product(&MatModQ::identity(x_i.u_i.len(), &x_i.A.q))
                      .mul_vec(z_i);

    Instance {
        A: x_i.A.clone(),
        t_i: updated_t,
        u_i: updated_u,
    }
}



impl Instance {
    fn t_i_add(&self, other: &VecModQ) -> VecModQ {
        if self.t_i.len() == other.len() {
            let mut out = Vec::with_capacity(self.t_i.len());
            for i in 0..self.t_i.len() {
                out.push(self.t_i.data[i].add_mod(&other.data[i]));
            }
            VecModQ::new(out, &self.t_i.q)
        } else {
            self.t_i.clone()
        }
    }
}

fn next_witness(x_i: &Instance, w_i: &Witness, c: &Challenge) -> Witness {
    let mut new_sj = Vec::new();

    for j in 0..w_i.s_j.len() {
        let C_t = construct_ct(c, w_i.s_j[j].len(), &x_i.A.q);
        let I_rj = MatModQ::identity(w_i.s_j[j].len(), &x_i.A.q);
        let updated_sj = C_t.tensor_product(&I_rj).mul_vec(&w_i.s_j[j]);
        new_sj.push(updated_sj);
    }

    let C_t = construct_ct(c, w_i.f.len(), &x_i.A.q);
    let updated_f = C_t.tensor_product(&MatModQ::identity(w_i.f.len(), &x_i.A.q))
                         .mul_vec(&w_i.f);

    Witness {
        s_j: new_sj,
        f: updated_f,
    }
}




fn add_vecmodq(a: &VecModQ, b: &VecModQ) -> VecModQ {
    if a.len() != b.len() {
        return a.clone();
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a.data[i].add_mod(&b.data[i]));
    }
    VecModQ::new(out, &a.q)
}

/// Generate a random challenge.
fn sample_challenge(q: &BigInt, dim: usize) -> Challenge {
    let mut rng = rand::thread_rng();
    let mut d = Vec::new();
    for _ in 0..dim {
        let r: i128 = rng.gen_range(-5..=5);
        d.push(ModQ::new(BigInt::from(r), q));
    }
    Challenge { data: VecModQ::new(d, q) }
}

//
// ========================================================
// 5) Single function combining "open" + checks (like paper)
// ========================================================

/// Now open_poly returns just a bool: true if all checks pass, false if fail.
/// This merges both "open" and "verification" into one function, matching
/// the paper's approach where if all rounds pass, we accept immediately.
fn open_poly(params: &CommitmentParams, comm: &CommitOutput, rounds: usize) -> bool {
    // 1) Check base commitment
    let base_ok = open_section4(params, comm);
    if !base_ok {
        return false;
    }

    // 2) Initialize instance and witness
    let mut x_i = Instance {
        A: params.A.clone(),
        t_i: comm.t.clone(),
        u_i: VecModQ::zero(params.A.rows, &params.q), // Initialize appropriately
    };
    let mut w_i = Witness {
        s_j: comm.s_arr.clone(),
        f: comm.f.clone(),
    };

    // 3) Run interactive argument for the specified number of rounds
    for _ in 0..rounds {
        // Prover computes z_i = (y_i, v_i)
        let z_i = prover_round_i(params, &x_i, &w_i);

        // Verifier checks
        let ok = verifier_round_i(params, &x_i, &z_i);
        if !ok {
            eprintln!("open_poly: Verification failed in a round");
            return false;
        }

        // Verifier issues challenge
        let c = sample_challenge(&params.q, x_i.t_i.len());

        // Both sides update instance and witness
        x_i = next_instance(&x_i, &z_i, &c);
        w_i = next_witness(&x_i, &w_i, &c);
    }

    // If all checks pass
    true
}


//
// ========================================================
// 6) MAIN
// ========================================================
fn main() {
    println!("=== Demo: single-phase open + checks (like the paper) ===");

    // 1) Choose a prime q and define A
    let q = BigInt::from(97u32);
    let rows = 2;
    let cols = 3;
    let mut rng = rand::thread_rng();

    // Build A (rows x cols)
    let mut dataA = Vec::new();
    for _ in 0..(rows * cols) {
        let r: i128 = rng.gen_range(0..10);
        dataA.push(ModQ::new(BigInt::from(r), &q));
    }
    let A = MatModQ::new(rows, cols, dataA, &q);

    // 2) Define expansions
    let r_vals = vec![2, 3]; // Initialize r_vals
    let ell = r_vals.len();  // Initialize ell as the length of r_vals

    let params = CommitmentParams {
        q: q.clone(),
        A,
        r_vals,
        ell,
        beta: 15,
        kappa: 2, // Define kappa appropriately
        tau: 3,   // Define tau appropriately
        n: 3,     // Define n appropriately
    };

    // Debug: Print params
    println!("A.rows: {}, A.cols: {}", params.A.rows, params.A.cols);
    println!("r_vals: {:?}, ell: {}", params.r_vals, params.ell);

    // 3) Define polynomial f (dimension = r_{ell+1} * A.cols => 3*3=9)
    let final_r = params.r_vals[params.r_vals.len() - 1];
    let fdim = final_r * params.A.cols;
    let mut fdata = Vec::new();
    for _ in 0..fdim {
        let rv = rng.gen_range(0..10);
        fdata.push(ModQ::new(BigInt::from(rv), &params.q));
    }
    let f = VecModQ::new(fdata, &params.q);

    // Debug: Print dimensions of f
    println!("f.dim: {}", fdim);

    // 4) Commit
    let comm_out = commit_section4(&params, &f);
    println!(
        "commit => t.len={}, s_arr.len={}",
        comm_out.t.len(),
        comm_out.s_arr.len()
    );

    // 5) Single open phase (includes checks) => "rounds" of interaction
    let rounds = 2;
    let success = open_poly(&params, &comm_out, rounds);
    println!("Protocol result => {}", success);
}

