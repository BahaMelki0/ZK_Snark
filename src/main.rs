use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_std::UniformRand;
use rand::thread_rng;
use rand::Rng;
use std::fmt;
use nalgebra::{DMatrix, DVector};

// ========================================================
// 1) Finite Field, Vectors, and Matrices Using Arkworks and Nalgebra
// ========================================================

// Type alias for field elements
type F = Fr;

// Vector over the finite field
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecF {
    pub data: DVector<F>,
}

impl VecF {
    pub fn new(data: DVector<F>) -> Self {
        VecF { data }
    }

    pub fn zero(len: usize) -> Self {
        VecF {
            data: DVector::from_element(len, F::zero()),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    // Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector lengths must match for addition."
        );
        let data: Vec<F> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        VecF {
            data: DVector::from_vec(data),
        }
    }

    // Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector lengths must match for multiplication."
        );
        let data: Vec<F> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a * *b)
            .collect();
        VecF {
            data: DVector::from_vec(data),
        }
    }

    // Scalar multiplication
    pub fn scalar_mul(&self, scalar: F) -> Self {
        VecF {
            data: self.data.clone() * scalar,
        }
    }

    /// Compute the maximum norm (max absolute value)
    pub fn norm(&self) -> F {
        self.data
            .iter()
            .cloned()
            .fold(F::zero(), |acc, x| if x > acc { x } else { acc })
    }
}

impl fmt::Display for VecF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements: Vec<String> = self.data.iter().map(|x| x.to_string()).collect();
        write!(f, "[{}]", elements.join(", "))
    }
}

// ========================================================
// Matrix over the finite field
// ========================================================

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatF {
    pub rows: usize,
    pub cols: usize,
    pub data: DMatrix<F>,
}

impl MatF {
    /// Creates a new matrix with specified dimensions and data
    pub fn new(rows: usize, cols: usize, data: Vec<F>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length ({}) must match rows * cols ({} * {}).",
            data.len(),
            rows,
            cols
        );
        MatF {
            rows,
            cols,
            data: DMatrix::from_row_slice(rows, cols, &data),
        }
    }

    /// Creates a zero matrix of given dimensions
    pub fn zero(rows: usize, cols: usize) -> Self {
        MatF {
            rows,
            cols,
            data: DMatrix::from_element(rows, cols, F::zero()),
        }
    }

    /// Creates an identity matrix of given size
    pub fn identity(size: usize) -> Self {
        MatF {
            rows: size,
            cols: size,
            data: DMatrix::identity(size, size),
        }
    }

    /// Custom matrix-vector multiplication
    pub fn custom_mul_vec(&self, v: &VecF) -> VecF {
        assert_eq!(
            self.cols,
            v.len(),
            "Matrix-vector multiplication dimension mismatch: matrix.cols = {}, vector.len = {}",
            self.cols,
            v.len()
        );

        let mut result_data = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let mut sum = F::zero();
            for c in 0..self.cols {
                sum += self.data[(r, c)] * v.data[c];
            }
            result_data.push(sum);
        }
        VecF::new(DVector::from_vec(result_data))
    }

    /// Computes the Kronecker product of two matrices
    pub fn kronecker_product(&self, other: &Self) -> Self {
        let kron = self.data.kronecker(&other.data);
        MatF {
            rows: self.rows * other.rows,
            cols: self.cols * other.cols,
            data: kron,
        }
    }

    /// Inverts a diagonal matrix (only works if matrix is diagonal)
    pub fn inverse_diagonal(&self) -> Option<Self> {
        if self.rows != self.cols {
            return None;
        }
        let mut inv_data = DMatrix::from_element(self.rows, self.cols, F::zero());
        for i in 0..self.rows {
            let elem = self.data[(i, i)];
            if elem.is_zero() {
                return None;
            }
            inv_data[(i, i)] = elem.inverse().unwrap();
        }
        Some(MatF {
            rows: self.rows,
            cols: self.cols,
            data: inv_data,
        })
    }

    /// Transposes the matrix
    pub fn transpose(&self) -> Self {
        MatF {
            rows: self.cols,
            cols: self.rows,
            data: self.data.transpose(),
        }
    }
}

impl fmt::Display for MatF {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.rows {
            let row: Vec<String> = (0..self.cols)
                .map(|c| self.data[(r, c)].to_string())
                .collect();
            writeln!(f, "[{}]", row.join(", "))?;
        }
        Ok(())
    }
}

// ========================================================
// 2) Protocol Structures and Utility Functions
// ========================================================

#[derive(Clone, Debug)]
pub struct CommitmentParams {
    pub q: F,               // Not used directly since F handles field operations
    pub a: MatF,            // Matrix a
    pub r_vals: Vec<usize>, // Vector of r values
    pub ell: usize,         // Number of rounds
    pub betas: Vec<F>,      // Beta values
    pub kappa: usize,       // Parameter kappa
    pub tau: usize,         // Parameter tau
    pub n: usize,           // Parameter n
}

#[derive(Clone, Debug)]
pub struct CommitOutput {
    pub t: VecF,
    pub s_arr: Vec<VecF>,
    pub f: VecF,
    pub u: VecF, // New field
}

#[derive(Clone)]
pub struct Instance {
    pub a: MatF, // Matrix a
    pub t_i: VecF,
    pub u_i: VecF,
}

#[derive(Clone)]
pub struct Witness {
    pub s_j: Vec<VecF>,
    pub f: VecF,
}

/// Minimal challenge structure.
#[derive(Clone)]
pub struct Challenge {
    pub data: VecF,
}

// ========================================================
// 3) Protocol Functions Using Arkworks
// ========================================================

/// Pads a vector to the target length with zeros.
fn pad_vector(v: &VecF, target_len: usize) -> VecF {
    if v.len() >= target_len {
        VecF::new(v.data.clone())
    } else {
        let zeros_to_append = target_len - v.len();
        let zeros = DVector::from_element(zeros_to_append, F::zero());
        let concatenated = DVector::from_iterator(
            v.len() + zeros_to_append,
            v.data.iter().chain(zeros.iter()).cloned(),
        );
        VecF::new(concatenated)
    }
}

/// Generates a diagonal gadget matrix with powers of two on the diagonal.
fn gadget_matrix(m: usize) -> MatF {
    let mut data = Vec::with_capacity(m * m);
    for r in 0..m {
        for c in 0..m {
            if r == c {
                // Compute 2^r in field F
                let val = F::from(2u64).pow(&[r as u64, 0, 0, 0]);
                data.push(val);
            } else {
                data.push(F::zero());
            }
        }
    }
    let matrix = MatF::new(m, m, data);
    println!("DEBUG: Gadget matrix (size {}x{}):\n{}", m, m, matrix);
    matrix
}

/// Applies the inverse of the gadget matrix to vector v
/// and returns the binary representation as a vector whose
/// entries are exactly 0 or 1 (encoded as elements of F).
fn g_inv_apply(_params: &CommitmentParams, v: &VecF, m: usize) -> VecF {
    let gadget = gadget_matrix(m);
    println!("DEBUG: Gadget matrix (before inversion):\n{}", gadget);

    let gadget_inv = gadget
        .inverse_diagonal()
        .expect("Gadget matrix is invertible.");
    println!("DEBUG: Inverted gadget matrix:\n{}", gadget_inv);

    let adjusted_v = if v.len() > gadget_inv.cols {
        println!(
            "DEBUG: Truncating vector v from length {} to {}",
            v.len(),
            gadget_inv.cols
        );
        VecF::new(v.data.rows_range(0..gadget_inv.cols).into_owned())
    } else if v.len() < gadget_inv.cols {
        println!(
            "DEBUG: Padding vector v from length {} to {}",
            v.len(),
            gadget_inv.cols
        );
        pad_vector(v, gadget_inv.cols)
    } else {
        v.clone()
    };
    println!("DEBUG: Adjusted v: {}", adjusted_v);

    let decoded = gadget_inv.custom_mul_vec(&adjusted_v);
    println!("DEBUG: Decoded (pre-rounding) vector: {}", decoded);

    let binary_data: Vec<F> = decoded.data.iter().map(|x| {
        if *x == F::zero() {
            F::zero()
        } else if *x == F::one() {
            F::one()
        } else {
            panic!("Non-binary value encountered in g_inv_apply: {}", x)
        }
    }).collect();

    let binary_vec = VecF::new(DVector::from_vec(binary_data));
    println!("DEBUG: Binary representation (g_inv_apply result): {}", binary_vec);
    binary_vec
}

/// Constructs the challenge matrix \( C_{i+1} \) as a column vector (dim × 1 matrix).
fn construct_ct(challenge: &Challenge, dim: usize, _a: &MatF) -> MatF {
    assert_eq!(
        challenge.data.len(),
        dim,
        "Challenge length must be exactly {} (got {})",
        dim,
        challenge.data.len()
    );
    MatF::new(dim, 1, challenge.data.data.iter().cloned().collect())
}

/// Generates a diagonal gadget matrix with powers of two on the diagonal.
/// The size of the matrix is `kappa * n` to align with protocol parameters.
fn generate_xj(_j: usize, kappa: usize, n: usize) -> MatF {
    let size = kappa * n;
    gadget_matrix(size)
}

/// Computes t_i by first decoding f to obtain y_i and then computing
/// t_i = (I_{m/kappa} ⊗ a) * y_i.
fn compute_t(params: &CommitmentParams, f: &VecF, m: usize) -> VecF {
    println!("Running compute_t");

    let y_i = g_inv_apply(params, f, m);
    println!("DEBUG: y_i = {}", y_i);

    let i_kappa_size = m / params.kappa;
    let i_kappa = MatF::identity(i_kappa_size);
    let tensor_i_kr_a = i_kappa.kronecker_product(&params.a);
    println!(
        "DEBUG: Kronecker product (I_{{m/kappa}} ⊗ a) dimensions: {}x{}",
        tensor_i_kr_a.rows, tensor_i_kr_a.cols
    );

    assert_eq!(
        y_i.len(),
        tensor_i_kr_a.cols,
        "y_i length ({}) must match tensor_i_kr_a.cols ({})",
        y_i.len(),
        tensor_i_kr_a.cols
    );

    let t_i = tensor_i_kr_a.custom_mul_vec(&y_i);
    println!("DEBUG: Computed t_i = {}", t_i);

    t_i
}

/// Computes u based on parameters and f (similar to compute_t).
fn compute_u(params: &CommitmentParams, f: &VecF) -> VecF {
    println!("Running compute_u");

    let m = params.kappa * params.n;
    let y_i = g_inv_apply(params, f, m);
    println!("DEBUG: y_i for u = {}", y_i);

    let i_kappa_size = m / params.kappa;
    let i_kappa = MatF::identity(i_kappa_size);
    let tensor_i_kr_a = i_kappa.kronecker_product(&params.a);
    println!(
        "DEBUG: Kronecker product (I_{{m/kappa}} ⊗ a) for u dimensions: {}x{}",
        tensor_i_kr_a.rows, tensor_i_kr_a.cols
    );

    assert_eq!(
        y_i.len(),
        tensor_i_kr_a.cols,
        "y_i length ({}) must match tensor_i_kr_a.cols ({}) for u",
        y_i.len(),
        tensor_i_kr_a.cols
    );

    let u_i = tensor_i_kr_a.custom_mul_vec(&y_i);
    println!("DEBUG: Computed u_i = {}", u_i);

    u_i
}

/// Computes s_j based on the current parameters, f, and j.
/// In this version we compute the encoded s_j and then decode it block‑wise.
fn compute_s_j(params: &CommitmentParams, f: &VecF, j: usize) -> VecF {
    println!("Running compute_s_j");

    let x_li = generate_xj(j, params.kappa, params.n);
    println!("DEBUG: Generated x_li for compute_s_j:\n{}", x_li);

    let i_kappa = MatF::identity(params.kappa);
    let tensor_i_kappa_x_li = i_kappa.kronecker_product(&x_li);
    println!(
        "DEBUG: Kronecker product (I_{{kappa}} ⊗ X_li) dimensions: {}x{}",
        tensor_i_kappa_x_li.rows, tensor_i_kappa_x_li.cols
    );

    // Adjust f to match the required column dimension.
    let f_padded = if f.len() < tensor_i_kappa_x_li.cols {
        println!(
            "DEBUG: Padding f from {} to {} to match tensor_i_kappa_x_li.cols",
            f.len(),
            tensor_i_kappa_x_li.cols
        );
        pad_vector(f, tensor_i_kappa_x_li.cols)
    } else if f.len() > tensor_i_kappa_x_li.cols {
        println!(
            "DEBUG: Truncating f from {} to {} to match tensor_i_kappa_x_li.cols",
            f.len(),
            tensor_i_kappa_x_li.cols
        );
        VecF::new(f.data.rows_range(0..tensor_i_kappa_x_li.cols).into_owned())
    } else {
        f.clone()
    };
    println!("DEBUG: f_padded for s_j: {}", f_padded);

    // Compute the encoded s_j.
    let s_encoded = tensor_i_kappa_x_li.custom_mul_vec(&f_padded);
    println!("DEBUG: Encoded s_j computed: {}", s_encoded);

    // Decode s_encoded block-wise.
    let block_len = f_padded.len() / params.kappa;
    let mut decoded_data = Vec::new();
    for block in 0..params.kappa {
        let start = block * block_len;
        let end = start + block_len;
        let block_vec = VecF::new(s_encoded.data.rows_range(start..end).into_owned());
        println!("DEBUG: s_encoded block {}: {}", block, block_vec);
        let decoded_block = g_inv_apply(params, &block_vec, block_len);
        println!("DEBUG: Decoded block {}: {}", block, decoded_block);
        // Extend with owned copies of the field elements.
        decoded_data.extend(decoded_block.data.iter().cloned());
    }
    let s_binary = VecF::new(DVector::from_vec(decoded_data));
    println!("DEBUG: Full decoded s_j (binary): {}", s_binary);

    // Optionally truncate if needed (here we truncate each block to 4 elements, if desired)
    let s_final = if s_binary.len() > params.kappa * 4 {
        println!("DEBUG: Truncating s_j binary to {} elements", params.kappa * 4);
        VecF::new(s_binary.data.rows_range(0..params.kappa * 4).into_owned())
    } else {
        s_binary
    };
    println!("DEBUG: Final s_j: {}", s_final);
    s_final
}

/// Commitment function.
fn commit_section4(params: &CommitmentParams, f: &VecF) -> CommitOutput {
    let m = params.kappa * params.n;
    let mut t_full = compute_t(params, f, m);
    if t_full.len() > 4 {
        println!("DEBUG: Truncating t from {} to 4", t_full.len());
        t_full = VecF::new(t_full.data.rows_range(0..4).into_owned());
    }

    let mut u_full = compute_u(params, f);
    if u_full.len() > 4 {
        println!("DEBUG: Truncating u from {} to 4", u_full.len());
        u_full = VecF::new(u_full.data.rows_range(0..4).into_owned());
    }

    let mut s_arr = Vec::new();
    for j in 0..params.ell {
        let s = compute_s_j(params, f, j);
        s_arr.push(s);
    }

    CommitOutput {
        t: t_full,
        s_arr,
        f: f.clone(),
        u: u_full,
    }
}

/// Verification of the commitment.
fn open_section4(params: &CommitmentParams, comm: &CommitOutput) -> bool {
    let m = params.kappa * params.n;
    let mut t_check = compute_t(params, &comm.f, m);
    if t_check.len() > 4 {
        t_check = VecF::new(t_check.data.rows_range(0..4).into_owned());
    }

    if t_check != comm.t {
        eprintln!("open_section4: mismatch t");
        return false;
    }

    for j in 0..comm.s_arr.len() {
        let mut sj_check = compute_s_j(params, &comm.f, j);
        if sj_check.len() > comm.s_arr[j].len() {
            sj_check = VecF::new(sj_check.data.rows_range(0..comm.s_arr[j].len()).into_owned());
        }
        if sj_check != comm.s_arr[j] {
            eprintln!("open_section4: mismatch s_j, j={}", j);
            return false;
        }
    }
    true
}

/// Prover's round function.
fn prover_round_i(
    params: &CommitmentParams,
    _x_i: &Instance,
    w_i: &Witness,
    _i: usize,
) -> VecF {
    println!("Running prover_round_i");

    let y_i = w_i.s_j[0].clone();
    let v_i = w_i.f.clone();

    let concatenated = DVector::from_iterator(
        y_i.data.len() + v_i.data.len(),
        y_i.data.iter().chain(v_i.data.iter()).cloned(),
    );
    VecF::new(concatenated)
}

/// Verifier's round function.
fn verifier_round_i(
    params: &CommitmentParams,
    x_i: &Instance,
    z_i: &VecF,
    c_i: &Challenge,
    _i: usize,
) -> bool {
    println!("\n=== Verifier Round {} ===", _i + 1);

    let r_i = params.r_vals[params.ell - _i - 1];
    println!("DEBUG: Current r_i = {}", r_i);

    let y_i_len = params.kappa * r_i;
    let y_i = VecF::new(z_i.data.rows_range(0..y_i_len).into_owned());
    let v_i = VecF::new(z_i.data.rows_range(y_i_len..).into_owned());

    println!("DEBUG: Extracted y_i = {}", y_i);
    println!("DEBUG: Extracted v_i = {}", v_i);

    // Check b_i,0: c_i * (G * y_i) == t_i
    let g = gadget_matrix(y_i.len());
    println!("DEBUG: Gadget matrix for b_i,0:\n{}", g);

    let g_y_i = g.custom_mul_vec(&y_i);
    println!("DEBUG: G * y_i = {}", g_y_i);

    assert_eq!(
        c_i.data.len(),
        g_y_i.len(),
        "Challenge length ({}) does not match G * y_i length ({})",
        c_i.data.len(),
        g_y_i.len()
    );
    let t_check = g_y_i.mul(&c_i.data);
    println!("DEBUG: Computed t_check = {}", t_check);

    let t_expected = x_i.t_i.clone();
    println!("DEBUG: Expected t_i = {}", t_expected);

    if t_check != VecF::new(t_expected.data.rows_range(0..4).into_owned()) {
        println!("DEBUG: b_i,0 check failed.");
        return false;
    } else {
        println!("DEBUG: b_i,0 check passed.");
    }    

    // Check b_i,1: (I_{kappa} ⊗ X_li) ⋅ v_i == u_i
    let x_li = generate_xj(params.ell - _i - 1, params.kappa, params.n);
    println!("DEBUG: Generated x_li matrix:\n{}", x_li);

    let i_kappa = MatF::identity(params.kappa);
    println!(
        "DEBUG: Identity matrix i_kappa (size {}x{})\n{}",
        params.kappa,
        params.kappa,
        i_kappa
    );

    let tensor_i_kappa_x_li = i_kappa.kronecker_product(&x_li);
    println!(
        "DEBUG: Kronecker product (I_{{kappa}} ⊗ X_li) dimensions: {}x{}",
        tensor_i_kappa_x_li.rows, tensor_i_kappa_x_li.cols
    );

    let mut v_i_padded = v_i.clone();
    if v_i.len() != tensor_i_kappa_x_li.cols {
        if v_i.len() < tensor_i_kappa_x_li.cols {
            println!(
                "DEBUG: Padding v_i from {} to {} to match tensor_i_kappa_x_li.cols",
                v_i.len(),
                tensor_i_kappa_x_li.cols
            );
            v_i_padded = pad_vector(&v_i, tensor_i_kappa_x_li.cols);
        } else {
            println!(
                "DEBUG: Truncating v_i from {} to {} to match tensor_i_kappa_x_li.cols",
                v_i.len(),
                tensor_i_kappa_x_li.cols
            );
            v_i_padded = VecF::new(
                v_i.data
                    .rows_range(0..tensor_i_kappa_x_li.cols)
                    .into_owned(),
            );
        }
    }
    println!("DEBUG: Adjusted v_i for b_i,1:\n{}", v_i_padded);

    let u_check = tensor_i_kappa_x_li.custom_mul_vec(&v_i_padded);
    println!("DEBUG: Computed u_check = {}", u_check);

    let u_expected = x_i.u_i.clone();
    println!("DEBUG: Expected u_i = {}", u_expected);

    if u_check != u_expected {
        println!("DEBUG: b_i,1 check failed.");
        return false;
    } else {
        println!("DEBUG: b_i,1 check passed.");
    }

    // Check b_i,2: Norm constraints
    let beta_li = params.betas[params.ell - _i - 1];
    let y_norm = y_i.norm();
    let v_norm = v_i.norm();

    let epsilon = F::from(1u64);

    println!(
        "DEBUG: Norm check: y_norm = {}, v_norm = {}, beta = {}, epsilon = {}",
        y_norm, v_norm, beta_li, epsilon
    );

    if y_norm > (beta_li + epsilon) || v_norm > (beta_li + epsilon) {
        println!("DEBUG: Norm constraints failed.");
        return false;
    } else {
        println!("DEBUG: Norm constraints passed.");
    }

    println!("DEBUG: verifier_round_i passed.");
    true
}

/// Generates a random challenge vector.
fn sample_challenge(dim: usize) -> Challenge {
    let mut rng = thread_rng();
    let mut d = Vec::with_capacity(dim);
    for _ in 0..dim {
        let r = F::rand(&mut rng);
        d.push(r);
    }
    Challenge {
        data: VecF::new(DVector::from_vec(d)),
    }
}

/// In this fixed version we generate a binary vector and gadget-encode it.
fn generate_random_f(dim: usize) -> VecF {
    let mut rng = thread_rng();
    let mut bits = Vec::with_capacity(dim);
    for _ in 0..dim {
        let bit: u64 = rng.gen_range(0..2);
        bits.push(F::from(bit));
    }
    let b_vec = VecF::new(DVector::from_vec(bits));
    println!("DEBUG: Generated binary vector: {}", b_vec);
    let gadget = gadget_matrix(dim);
    let f = gadget.custom_mul_vec(&b_vec);
    println!("DEBUG: Gadget-encoded f: {}", f);
    f
}

fn next_instance(
    x_i: &Instance,
    z_i: &VecF,
    c: &Challenge,
    params: &CommitmentParams,
    _i: usize,
) -> Instance {
    let r_i = params.r_vals[params.ell - _i - 1];
    let y_i_len = params.kappa * r_i;
    let y_i = VecF::new(z_i.data.rows_range(0..y_i_len).into_owned());
    let v_i = VecF::new(z_i.data.rows_range(y_i_len..).into_owned());
    
    let c_t = construct_ct(c, y_i.len(), &x_i.a);
    let g = gadget_matrix(y_i.len());
    let g_y_i = g.custom_mul_vec(&y_i);
    let c_t_vec = VecF::new(c_t.data.column(0).into_owned());

    println!("DEBUG: c_t_vec.len() = {}", c_t_vec.len());
    println!("DEBUG: g_y_i.len() = {}", g_y_i.len());

    let t_next = c_t_vec.mul(&g_y_i);
    println!("DEBUG: t_next = {}", t_next);

    let mut u_next = DVector::zeros(v_i.len());
    let block_size = c_t_vec.len();

    for block in 0..params.kappa {
        let start = block * block_size;
        let end = start + block_size;
        let v_block = v_i.data.rows_range(start..end).into_owned();
        let v_block_vec = VecF::new(v_block);

        assert_eq!(
            v_block_vec.len(),
            c_t_vec.len(),
            "v_block_vec.len() ({}) does not match c_t_vec.len() ({})",
            v_block_vec.len(),
            c_t_vec.len()
        );

        let updated_block = c_t_vec.mul(&v_block_vec);
        println!("DEBUG: updated_block u_i[{}..{}] = {}", start, end, updated_block);
        u_next.rows_range_mut(start..end).copy_from(&updated_block.data);
    }

    let u_next_vec = VecF::new(u_next);
    println!("DEBUG: u_next = {}", u_next_vec);

    Instance {
        a: x_i.a.clone(),
        t_i: t_next,
        u_i: u_next_vec,
    }
}

fn next_witness(params: &CommitmentParams, w_i: &Witness, c: &Challenge) -> Witness {
    let mut new_sj = Vec::new();

    for j in 0..w_i.s_j.len() {
        let c_t = construct_ct(c, w_i.s_j[j].len(), &params.a);
        let r_scale = params.kappa;
        let i = MatF::identity(r_scale);
        let c_t_transposed = c_t.transpose();

        println!(
            "DEBUG: c_t_transposed dimensions: {}x{}",
            c_t_transposed.rows, c_t_transposed.cols
        );
        println!("DEBUG: s_j[j].len() = {}", w_i.s_j[j].len());

        assert_eq!(
            c_t_transposed.cols,
            i.cols,
            "c_t_transposed.cols ({}) does not match i.cols ({})",
            c_t_transposed.cols,
            i.cols
        );

        let updated_sj = c_t_transposed.kronecker_product(&i).custom_mul_vec(&w_i.s_j[j]);
        println!("DEBUG: updated_sj = {}", updated_sj);
        new_sj.push(updated_sj);
    }

    let c_t = construct_ct(c, w_i.f.len() / params.kappa, &params.a);
    let r_scale_f = params.kappa;
    let i_f = MatF::identity(r_scale_f);

    println!(
        "DEBUG: c_t dimensions for f update: {}x{}",
        c_t.rows, c_t.cols
    );
    println!("DEBUG: i_f dimensions: {}x{}", i_f.rows, i_f.cols);

    let c_t_vec: VecF = VecF::new(c_t.data.column(0).into_owned());

    println!("DEBUG: c_t_vec.len() = {}", c_t_vec.len());
    println!("DEBUG: w_i.f.len() = {}", w_i.f.len());

    let mut updated_f = DVector::zeros(w_i.f.len());
    let block_size = c_t_vec.len();

    for block in 0..params.kappa {
        let start = block * block_size;
        let end = start + block_size;
        let f_block = w_i.f.data.rows_range(start..end).into_owned();
        let f_block_vec = VecF::new(f_block);

        assert_eq!(
            f_block_vec.len(),
            c_t_vec.len(),
            "f_block_vec.len() ({}) does not match c_t_vec.len() ({})",
            f_block_vec.len(),
            c_t_vec.len()
        );

        let updated_block = c_t_vec.mul(&f_block_vec);
        println!("DEBUG: updated_block[{}..{}] = {}", start, end, updated_block);
        updated_f.rows_range_mut(start..end).copy_from(&updated_block.data);
    }

    println!("DEBUG: updated_f = {}", VecF::new(updated_f.clone()));

    Witness {
        s_j: new_sj,
        f: VecF::new(updated_f),
    }
}

/// Open_poly returns true if all checks pass.
fn open_poly(params: &CommitmentParams, comm: &CommitOutput, rounds: usize) -> bool {
    let base_ok = open_section4(params, comm);
    if !base_ok {
        return false;
    }

    let mut x_i = Instance {
        a: params.a.clone(),
        t_i: comm.t.clone(),
        u_i: comm.u.clone(),
    };

    println!("DEBUG: Initializing instance x_i");
    println!("DEBUG: x_i.a =\n{}", params.a);
    println!("DEBUG: x_i.t_i = {}", comm.t);
    println!("DEBUG: x_i.u_i = {}", x_i.u_i);

    let mut w_i = Witness {
        s_j: comm.s_arr.clone(),
        f: comm.f.clone(),
    };

    for i in 0..rounds {
        println!("=== Round {} ===", i + 1);

        let z_i = prover_round_i(params, &x_i, &w_i, i);
        println!("Prover's z_i: {}", z_i);

        let r_i = params.r_vals[params.ell - i - 1];
        println!("DEBUG: Current r_i for round {}: {}", i + 1, r_i);

        let c = sample_challenge(params.kappa * r_i);
        println!("Verifier's challenge (len={}): {}", c.data.len(), c.data);

        let ok = verifier_round_i(params, &x_i, &z_i, &c, i);
        if !ok {
            eprintln!("open_poly: Verification failed in round {}", i + 1);
            return false;
        }

        x_i = next_instance(&x_i, &z_i, &c, params, i);
        w_i = next_witness(params, &w_i, &c);
    }

    true
}

/// In this fixed version we generate a binary vector and gadget-encode it.
fn main() {
    println!("=== Demo: ZK Proof System with Arkworks ===");

    let a = MatF::identity(2);

    let params = CommitmentParams {
        q: F::one(),
        a: a.clone(),
        r_vals: vec![4, 2],
        ell: 2,
        betas: vec![
            F::from(1_800_000_000_000_000_000u64),
            F::from(1_800_000_000_000_000_000u64),
        ],
        kappa: 2,
        tau: 1,
        n: 4,
    };

    let fdim = params.kappa * params.n;
    let f_vec = generate_random_f(fdim);

    println!("Generated f (gadget-encoded): {}", f_vec);

    let comm_out = commit_section4(&params, &f_vec);
    println!(
        "commit => t.len={}, s_arr.len={}, u.len={}",
        comm_out.t.len(),
        comm_out.s_arr.len(),
        comm_out.u.len()
    );

    let w_i = Witness {
        s_j: comm_out.s_arr.clone(),
        f: comm_out.f.clone(),
    };

    let protocol_success = open_poly(&params, &comm_out, params.ell);

    if protocol_success && open_section4(&params, &comm_out) {
        println!("Protocol result => true");
    } else {
        println!("Protocol result => false");
    }
}
