# Lattice-Based Polynomial Commitment in Rust

Implementation of the polynomial commitment scheme from:

> **"Polynomial Commitments from Lattices: Post-Quantum Security, Fast Verification and Transparent Setup"**  
> Cini, Malavolta, Nguyen, Wee — ePrint 2024/281

Covers **Section 4** (tree-based SIS commitment) and **Section 5 / Figure 6** (recursive folding protocol) in ~500 lines of safe Rust with no cryptographic library dependencies beyond `rand`.

---

## What's implemented

### Section 4 — Tree-based SIS commitment

The commitment is built as an ℓ-level Merkle-like tree over SIS (Shortest Integer Solution) instances.

```
s_{ℓ-1} = G⁻¹(f)
s_j     = G⁻¹( (I_{r^{j+1}·κ} ⊗ A) · s_{j+1} )   j = ℓ-2 … 0
t       = (I_κ ⊗ A) · s_0
```

- **`gadget(m, α, δ)`** — gadget matrix G_{m,α,δ} of shape m × (m·α); row i holds [δ⁰, δ¹, …, δ^{α-1}] at columns i·α … (i+1)·α−1
- **`g_inv(v)`** — base-δ little-endian digit decomposition; satisfies G · G⁻¹(v) = v mod q
- **`commit(params, f)`** — builds the full tree bottom-up
- **`verify_commit(params, comm)`** — checks the three recurrence relations and norm bounds

### Section 5 / Figure 6 — Recursive folding protocol

Each of the ℓ rounds folds the instance by a factor of rκ using a binary challenge matrix C ∈ {0,1}^{rκ × κ}.

**Prover sends** `(y_i, v_i)` where:
- `y_i = s^{(i)}_0` — current top opening vector
- `v_i = eval_witness(f^{(i)}, x[0..ℓ-i-2])` — partial polynomial evaluation

**Verifier checks** three conditions:

| Check | Formula |
|-------|---------|
| b₀ | `(I_κ ⊗ A) · y_i = t^{(i)}` |
| b₁ | `‖y_i‖_∞ ≤ β_i = (rκ)^i · (δ−1)` |
| b₂ | `(I_{κn} ⊗ x[ℓ-1-i]) · v_i = u^{(i)}` |

**State transitions** after challenge C_{i+1}:
```
t^{(i+1)} = (C^T ⊗ I_n) · G_{rκn} · y_i
u^{(i+1)} = (C^T ⊗ I_n) · v_i
s^{(i+1)}_j = (C^T ⊗ I_{r^{j+1}·n·α}) · s^{(i)}_{j+1}
f^{(i+1)}   = (C^T ⊗ I_{r^{depth-1}·n}) · f^{(i)}
```

---

## Parameters

| Symbol | Meaning |
|--------|---------|
| q | ring modulus |
| n | SIS matrix height |
| r | folding factor |
| κ | statistical security parameter |
| α = ⌈log_δ q⌉ | gadget digits per element |
| δ | gadget base (typically 2) |
| ℓ | tree depth = number of rounds |
| A ∈ Z_q^{n × rnα} | SIS matrix |
| x[j] ∈ Z_q^r | evaluation vector for level j |

---

## Getting started

```sh
# Clone
git clone https://github.com/BahaMelki0/ZK_Snark.git
cd ZK_Snark

# Run the demo  (toy parameters: q=8, δ=2, α=3, n=2, r=2, κ=1, ℓ=2)
cargo run

# Run all 21 tests
cargo test
```

Expected output:

```
=== Lattice Polynomial Commitment Demo ===
n=2 r=2 κ=1 α=3 δ=2 ℓ=2 q=8
eval points x = [[1, 3], [1, 1]]
f (len=8)  = [...]
t = [...]
u = [...]
verify_commit → true
run_protocol  → true
All checks passed.
```

---

## Test coverage

```
gadget_structure_2_3_2      — gadget matrix shape and values
g_inv_roundtrip             — G · G⁻¹(v) = v for all single-element inputs
g_inv_multi                 — G · G⁻¹(v) = v for multi-element vectors
g_inv_digits_in_range       — all digits ∈ {0, …, δ-1}
eval_witness_ell1_trivial   — depth-0: eval_witness(f, []) = f
eval_witness_ell2_dimension — depth-1: output length = rκn
eval_witness_linearity      — eval is linear in f mod q
b2_holds_at_commit          — u = (I_{κn} ⊗ x[ℓ-1]) · v₀ at commit time
b2_folds_correctly          — fold invariant holds after one challenge
commit_verify_ell1/2/3      — honest commitment passes for ℓ = 1, 2, 3
tampered_t/f/s0_rejected    — tampered openings are rejected
protocol_honest_ell1/2/3    — honest prover passes all three checks
protocol_tampered_t/u_fails — wrong t or u fails the protocol
kron_mixed_product          — (A⊗B)(C⊗D) = (AC)⊗(BD)
```

---

## Dependencies

```toml
[dependencies]
rand = "0.8"
```

No finite-field library, no matrix library — all arithmetic is plain `i64` mod q.

---

## Reference

Cini, V., Malavolta, G., Nguyen, N. K., & Wee, H. (2024).  
*Polynomial Commitments from Lattices: Post-Quantum Security, Fast Verification and Transparent Setup.*  
Cryptology ePrint Archive, Report 2024/281.  
https://eprint.iacr.org/2024/281
