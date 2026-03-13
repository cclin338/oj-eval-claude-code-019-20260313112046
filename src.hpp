#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // Concatenate all keys K[0]...K[i] into K_all [i+1, d]
    Matrix* K_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (K_all == nullptr) {
        K_all = matrix_memory_allocator.Allocate("K_all_" + std::to_string(i));
        gpu_sim.Copy(keys[j], K_all, kInGpuHbm);
      } else {
        Matrix* new_K_all = matrix_memory_allocator.Allocate("K_all_temp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(K_all, keys[j], new_K_all, 0, kInGpuHbm);
        K_all = new_K_all;
      }
    }

    // Concatenate all values V[0]...V[i] into V_all [i+1, d]
    Matrix* V_all = nullptr;
    for (size_t j = 0; j <= i; ++j) {
      if (V_all == nullptr) {
        V_all = matrix_memory_allocator.Allocate("V_all_" + std::to_string(i));
        gpu_sim.Copy(values[j], V_all, kInGpuHbm);
      } else {
        Matrix* new_V_all = matrix_memory_allocator.Allocate("V_all_temp_" + std::to_string(i) + "_" + std::to_string(j));
        gpu_sim.Concat(V_all, values[j], new_V_all, 0, kInGpuHbm);
        V_all = new_V_all;
      }
    }

    // Move Q, K_all, V_all to SRAM for faster computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K_all);
    gpu_sim.MoveMatrixToSharedMem(V_all);

    // Transpose K_all to get K_all^T [d, i+1]
    gpu_sim.Transpose(K_all, kInSharedMemory);

    // Compute Q @ K_all^T = [i+1, d] @ [d, i+1] = [i+1, i+1]
    Matrix* QK = matrix_memory_allocator.Allocate("QK_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_all, QK);

    // Apply row-wise Softmax: compute each row independently and concat
    Matrix* softmax_QK = nullptr;
    for (size_t row = 0; row <= i; ++row) {
      // Get row from QK
      Matrix* qk_row = matrix_memory_allocator.Allocate("qk_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.GetRow(QK, row, qk_row, kInSharedMemory);

      // Compute exp of row
      Matrix* exp_row = matrix_memory_allocator.Allocate("exp_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.MatExp(qk_row, exp_row);

      // Sum elements in this row
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.Sum(exp_row, row_sum);

      // Divide row by sum to get softmax
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(row));
      gpu_sim.MatDiv(exp_row, row_sum, softmax_row);

      // Concatenate to build softmax_QK
      if (softmax_QK == nullptr) {
        softmax_QK = softmax_row;
      } else {
        Matrix* new_softmax = matrix_memory_allocator.Allocate("softmax_temp_" + std::to_string(i) + "_" + std::to_string(row));
        gpu_sim.Concat(softmax_QK, softmax_row, new_softmax, 0, kInSharedMemory);
        softmax_QK = new_softmax;
      }
    }

    // Compute attention_weights @ V_all = [i+1, i+1] @ [i+1, d] = [i+1, d]
    Matrix* result = matrix_memory_allocator.Allocate("result_" + std::to_string(i));
    gpu_sim.MatMul(softmax_QK, V_all, result);

    // Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu