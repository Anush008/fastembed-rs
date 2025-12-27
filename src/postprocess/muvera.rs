//! MUVERA (Multi-Vector Retrieval Architecture) implementation.
//!
//! Converts variable-length multi-vector embeddings into fixed-dimensional encodings
//! using SimHash clustering and random projections.

use anyhow::Result;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::LateInteractionTextEmbedding;

/// Maximum Hamming distance value (64 bits + 1)
const MAX_HAMMING_DISTANCE: u32 = 65;

/// Precomputed popcount lookup table for bytes
const POPCOUNT_LUT: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = (i as u8).count_ones() as u8;
        i += 1;
    }
    table
};

/// Compute Hamming distance between two u64 values
#[inline]
fn hamming_distance(a: u64, b: u64) -> u32 {
    let xor = a ^ b;
    let bytes = xor.to_ne_bytes();
    bytes.iter().map(|&b| POPCOUNT_LUT[b as usize] as u32).sum()
}

/// Compute full Hamming distance matrix for cluster IDs 0..n
fn hamming_distance_matrix(n: usize) -> Vec<Vec<u32>> {
    let mut matrix = vec![vec![0u32; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = hamming_distance(i as u64, j as u64);
        }
    }
    matrix
}

/// SimHash projection component for MUVERA clustering.
///
/// Uses random hyperplanes to partition the vector space into 2^k_sim clusters.
#[derive(Debug, Clone)]
pub struct SimHashProjection {
    /// Random hyperplane normal vectors of shape (dim, k_sim)
    simhash_vectors: Vec<Vec<f32>>,
    k_sim: usize,
    dim: usize,
}

impl SimHashProjection {
    /// Create a new SimHash projection with random hyperplanes.
    ///
    /// # Arguments
    /// * `k_sim` - Number of SimHash functions (creates 2^k_sim clusters)
    /// * `dim` - Dimensionality of input vectors
    /// * `rng` - Random number generator for reproducibility
    pub fn new(k_sim: usize, dim: usize, rng: &mut impl Rng) -> Self {
        // Generate k_sim random hyperplanes from standard normal distribution
        // Shape: (dim, k_sim) - each column is a hyperplane normal vector
        let mut simhash_vectors = vec![vec![0.0f32; k_sim]; dim];
        for row in simhash_vectors.iter_mut().take(dim) {
            for cell in row.iter_mut().take(k_sim) {
                *cell = StandardNormal.sample(rng);
            }
        }

        Self {
            simhash_vectors,
            k_sim,
            dim,
        }
    }

    /// Compute cluster IDs for a batch of vectors using SimHash.
    ///
    /// # Arguments
    /// * `vectors` - Input vectors of shape (n_vectors, dim)
    ///
    /// # Returns
    /// Vector of cluster IDs in range [0, 2^k_sim - 1]
    pub fn get_cluster_ids(&self, vectors: &[Vec<f32>]) -> Vec<u64> {
        vectors
            .iter()
            .map(|vec| {
                assert_eq!(vec.len(), self.dim, "Vector dimension mismatch");

                // Compute dot product with each hyperplane: vec @ simhash_vectors
                // Result shape: (k_sim,)
                let mut dot_products = vec![0.0f32; self.k_sim];
                for (d, &value) in vec.iter().enumerate().take(self.dim) {
                    for (k, dp) in dot_products.iter_mut().enumerate().take(self.k_sim) {
                        *dp += value * self.simhash_vectors[d][k];
                    }
                }

                // Convert signs to cluster ID: (dot_product > 0) @ (1 << arange(k_sim))
                let mut cluster_id = 0u64;
                for (k, &dp) in dot_products.iter().enumerate().take(self.k_sim) {
                    if dp > 0.0 {
                        cluster_id |= 1u64 << k;
                    }
                }

                cluster_id
            })
            .collect()
    }
}

/// MUVERA (Multi-Vector Retrieval Architecture) algorithm implementation.
///
/// Creates Fixed Dimensional Encodings (FDEs) from variable-length sequences
/// of vectors using SimHash clustering and random projections.
#[derive(Debug, Clone)]
pub struct Muvera {
    /// Number of SimHash functions per projection
    k_sim: usize,
    /// Input vector dimensionality
    dim: usize,
    /// Output dimensionality after random projection
    dim_proj: usize,
    /// Number of random projection repetitions
    r_reps: usize,
    /// SimHash projections for each repetition
    simhash_projections: Vec<SimHashProjection>,
    /// Random projection matrices: (r_reps, dim, dim_proj) with values in {-1, +1}
    dim_reduction_projections: Vec<Vec<Vec<f32>>>,
    /// Precomputed Hamming distance matrix for cluster centers
    hamming_matrix: Vec<Vec<u32>>,
    /// Number of partitions (2^k_sim)
    num_partitions: usize,
}

impl Muvera {
    /// Create a new MUVERA instance.
    ///
    /// # Arguments
    /// * `dim` - Dimensionality of individual input vectors
    /// * `k_sim` - Number of SimHash functions (creates 2^k_sim clusters). Default: 5
    /// * `dim_proj` - Dimensionality after random projection (must be <= dim). Default: 16
    /// * `r_reps` - Number of random projection repetitions. Default: 20
    /// * `random_seed` - Seed for random number generator. Default: 42
    ///
    /// # Errors
    /// Returns error if dim_proj > dim
    pub fn new(
        dim: usize,
        k_sim: Option<usize>,
        dim_proj: Option<usize>,
        r_reps: Option<usize>,
        random_seed: Option<u64>,
    ) -> Result<Self> {
        let k_sim = k_sim.unwrap_or(5);
        let dim_proj = dim_proj.unwrap_or(16);
        let r_reps = r_reps.unwrap_or(20);
        let random_seed = random_seed.unwrap_or(42);

        if dim_proj > dim {
            return Err(anyhow::anyhow!(
                "Cannot project to higher dimensionality (dim_proj={} > dim={})",
                dim_proj,
                dim
            ));
        }

        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(random_seed);

        // Create r_reps independent SimHash projections
        let simhash_projections: Vec<SimHashProjection> = (0..r_reps)
            .map(|_| SimHashProjection::new(k_sim, dim, &mut rng))
            .collect();

        // Create random projection matrices with entries from {-1, +1} (Rademacher)
        let dim_reduction_projections: Vec<Vec<Vec<f32>>> = (0..r_reps)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        (0..dim_proj)
                            .map(|_| if rng.gen::<bool>() { 1.0f32 } else { -1.0f32 })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let num_partitions = 1 << k_sim;
        let hamming_matrix = hamming_distance_matrix(num_partitions);

        Ok(Self {
            k_sim,
            dim,
            dim_proj,
            r_reps,
            simhash_projections,
            dim_reduction_projections,
            hamming_matrix,
            num_partitions,
        })
    }

    /// Create a Muvera instance from a late interaction embedding model.
    ///
    /// This extracts the embedding dimension from the model automatically.
    pub fn from_late_interaction_model(
        model: &LateInteractionTextEmbedding,
        k_sim: Option<usize>,
        dim_proj: Option<usize>,
        r_reps: Option<usize>,
        random_seed: Option<u64>,
    ) -> Result<Self> {
        Self::new(model.dim(), k_sim, dim_proj, r_reps, random_seed)
    }

    /// Get the output embedding size.
    pub fn embedding_size(&self) -> usize {
        self.r_reps * self.num_partitions * self.dim_proj
    }

    /// Get the number of SimHash functions (k_sim parameter)
    pub fn k_sim(&self) -> usize {
        self.k_sim
    }

    /// Process a document's vectors into a Fixed Dimensional Encoding (FDE).
    ///
    /// Uses document-specific settings: normalizes cluster centers by vector count
    /// and fills empty clusters using Hamming distance-based selection.
    ///
    /// # Arguments
    /// * `vectors` - Document vectors of shape (n_tokens, dim)
    ///
    /// # Returns
    /// Fixed dimensional encoding of length (r_reps * 2^k_sim * dim_proj)
    pub fn process_document(&self, vectors: &[Vec<f32>]) -> Vec<f32> {
        self.process(vectors, true, true)
    }

    /// Process a query's vectors into a Fixed Dimensional Encoding (FDE).
    ///
    /// Uses query-specific settings: no normalization by count and no empty
    /// cluster filling to preserve query vector magnitudes.
    ///
    /// # Arguments
    /// * `vectors` - Query vectors of shape (n_tokens, dim)
    ///
    /// # Returns
    /// Fixed dimensional encoding of length (r_reps * 2^k_sim * dim_proj)
    pub fn process_query(&self, vectors: &[Vec<f32>]) -> Vec<f32> {
        self.process(vectors, false, false)
    }

    /// Core processing method.
    ///
    /// # Arguments
    /// * `vectors` - Input vectors of shape (n_vectors, dim)
    /// * `fill_empty_clusters` - Whether to fill empty clusters using nearest vectors
    /// * `normalize_by_count` - Whether to normalize cluster centers by count
    pub fn process(
        &self,
        vectors: &[Vec<f32>],
        fill_empty_clusters: bool,
        normalize_by_count: bool,
    ) -> Vec<f32> {
        assert!(
            vectors.iter().all(|v| v.len() == self.dim),
            "All vectors must have dimension {}",
            self.dim
        );

        let mut output = Vec::with_capacity(self.embedding_size());

        for proj_idx in 0..self.r_reps {
            let simhash = &self.simhash_projections[proj_idx];

            // Initialize cluster centers and track which vectors belong to each cluster
            let mut cluster_centers = vec![vec![0.0f32; self.dim]; self.num_partitions];
            let mut cluster_vector_indices: Vec<Vec<usize>> = vec![Vec::new(); self.num_partitions];

            // Assign vectors to clusters and accumulate centers (sum)
            let cluster_ids = simhash.get_cluster_ids(vectors);
            for (vec_idx, &cluster_id) in cluster_ids.iter().enumerate() {
                let cluster_idx = cluster_id as usize;
                for d in 0..self.dim {
                    cluster_centers[cluster_idx][d] += vectors[vec_idx][d];
                }
                cluster_vector_indices[cluster_idx].push(vec_idx);
            }

            // Compute cluster counts and empty mask
            let cluster_counts: Vec<usize> =
                cluster_vector_indices.iter().map(|v| v.len()).collect();
            let empty_mask: Vec<bool> = cluster_counts.iter().map(|&c| c == 0).collect();

            // Normalize by count if requested (only non-empty clusters)
            if normalize_by_count {
                for (cluster_idx, &count) in cluster_counts.iter().enumerate() {
                    if count > 0 {
                        for val in cluster_centers[cluster_idx].iter_mut() {
                            *val /= count as f32;
                        }
                    }
                }
            }

            // Fill empty clusters using Hamming distance
            if fill_empty_clusters {
                // For each cluster (row i), find the nearest NON-EMPTY cluster (column j)
                // by masking empty columns with MAX_HAMMING_DISTANCE
                // This matches Python: masked_hamming = np.where(empty_mask[None, :], MAX, hamming)
                // Then argmin along axis=1
                let mut nearest_non_empty: Vec<usize> = vec![0; self.num_partitions];
                for (i, nearest) in nearest_non_empty.iter_mut().enumerate() {
                    let mut min_dist = MAX_HAMMING_DISTANCE;
                    let mut best_j = 0usize;
                    for (j, &is_empty) in empty_mask.iter().enumerate() {
                        let dist = if is_empty {
                            MAX_HAMMING_DISTANCE
                        } else {
                            self.hamming_matrix[i][j]
                        };
                        if dist < min_dist {
                            min_dist = dist;
                            best_j = j;
                        }
                    }
                    *nearest = best_j;
                }

                // Now fill empty clusters: for each empty cluster i,
                // use first vector from nearest_non_empty[i]
                for cluster_idx in 0..self.num_partitions {
                    if empty_mask[cluster_idx] {
                        let source_cluster = nearest_non_empty[cluster_idx];
                        if !cluster_vector_indices[source_cluster].is_empty() {
                            let fill_vec_idx = cluster_vector_indices[source_cluster][0];
                            cluster_centers[cluster_idx] = vectors[fill_vec_idx].clone();
                        }
                    }
                }
            }

            // Apply random projection for dimensionality reduction
            let dim_reduction = &self.dim_reduction_projections[proj_idx];
            let scale = 1.0 / (self.dim_proj as f32).sqrt();

            for center in cluster_centers.iter().take(self.num_partitions) {
                if self.dim_proj < self.dim {
                    // Project: (1/sqrt(dim_proj)) * (cluster_center @ projection_matrix)
                    let mut projected = vec![0.0f32; self.dim_proj];
                    for (d, row) in dim_reduction.iter().enumerate().take(self.dim) {
                        for (p, slot) in projected.iter_mut().enumerate() {
                            *slot += center[d] * row[p];
                        }
                    }
                    output.extend(projected.into_iter().map(|v| v * scale));
                } else {
                    // No projection needed (dim_proj == dim)
                    output.extend_from_slice(center);
                }
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(0, 1), 1);
        assert_eq!(hamming_distance(0b1111, 0b0000), 4);
        assert_eq!(hamming_distance(0b1010, 0b0101), 4);
    }

    #[test]
    fn test_muvera_output_size() {
        let muvera = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
        // r_reps * 2^k_sim * dim_proj = 20 * 32 * 16 = 10240
        assert_eq!(muvera.embedding_size(), 20 * 32 * 16);
    }

    #[test]
    fn test_muvera_process() {
        let muvera = Muvera::new(128, Some(4), Some(8), Some(10), Some(42)).unwrap();

        // Create some test vectors
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let doc_encoding = muvera.process_document(&vectors);
        let query_encoding = muvera.process_query(&vectors);

        assert_eq!(doc_encoding.len(), muvera.embedding_size());
        assert_eq!(query_encoding.len(), muvera.embedding_size());
    }

    #[test]
    fn test_dim_proj_validation() {
        let result = Muvera::new(128, Some(5), Some(256), Some(20), Some(42));
        assert!(result.is_err());
    }

    #[test]
    fn test_muvera_deterministic() {
        let muvera1 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
        let muvera2 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let fde1 = muvera1.process_document(&vectors);
        let fde2 = muvera2.process_document(&vectors);

        assert_eq!(fde1, fde2, "MUVERA should be deterministic with same seed");
    }

    #[test]
    fn test_muvera_different_seeds() {
        let muvera1 = Muvera::new(128, Some(5), Some(16), Some(20), Some(42)).unwrap();
        let muvera2 = Muvera::new(128, Some(5), Some(16), Some(20), Some(123)).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect())
            .collect();

        let fde1 = muvera1.process_document(&vectors);
        let fde2 = muvera2.process_document(&vectors);

        assert_ne!(
            fde1, fde2,
            "Different seeds should produce different results"
        );
    }

    #[test]
    fn test_empty_cluster_filling() {
        // Test with very few vectors to ensure some clusters are empty
        let muvera = Muvera::new(8, Some(3), Some(4), Some(2), Some(42)).unwrap();

        // Only 2 vectors, but 2^3 = 8 clusters, so most will be empty
        let vectors: Vec<Vec<f32>> = vec![vec![1.0; 8], vec![-1.0; 8]];

        let doc_encoding = muvera.process_document(&vectors);

        // Should not panic and should produce valid output
        assert_eq!(doc_encoding.len(), muvera.embedding_size());

        // Verify no NaN or Inf values
        assert!(doc_encoding.iter().all(|&v| v.is_finite()));
    }
}
