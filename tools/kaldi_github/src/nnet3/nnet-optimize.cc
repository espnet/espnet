// nnet3/nnet-optimize.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015  Xiaohui Zhang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iomanip>
#include "nnet3/nnet-optimize.h"
#include "nnet3/nnet-optimize-utils.h"
#include "base/timer.h"

namespace kaldi {
namespace nnet3 {

void NnetOptimizeOptions::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<NnetOptimizeOptions>");
  ExpectToken(is, binary, "<Optimize>");
  ReadBasicType(is, binary, &optimize);
  ExpectToken(is, binary, "<ConsolidateModelUpdate>");
  ReadBasicType(is, binary, &consolidate_model_update);
  ExpectToken(is, binary, "<PropagateInPlace>");
  ReadBasicType(is, binary, &propagate_in_place);
  ExpectToken(is, binary, "<BackpropInPlace>");
  ReadBasicType(is, binary, &backprop_in_place);
  if (PeekToken(is, binary) == 'O') {
    ExpectToken(is, binary, "<OptimizeRowOps>");
    ReadBasicType(is, binary, &optimize_row_ops);
  }
  if (PeekToken(is, binary) == 'S') {
    ExpectToken(is, binary, "<SplitRowOps>");
    ReadBasicType(is, binary, &split_row_ops);
  }
  if (PeekToken(is, binary) == 'E') {
    ExpectToken(is, binary, "<ExtendMatrices>");
    ReadBasicType(is, binary, &extend_matrices);
  }
  ExpectToken(is, binary, "<ConvertAddition>");
  ReadBasicType(is, binary, &convert_addition);
  ExpectToken(is, binary, "<RemoveAssignments>");
  ReadBasicType(is, binary, &remove_assignments);
  ExpectToken(is, binary, "<AllowLeftMerge>");
  ReadBasicType(is, binary, &allow_left_merge);
  ExpectToken(is, binary, "<AllowRightMerge>");
  ReadBasicType(is, binary, &allow_right_merge);
  ExpectToken(is, binary, "<InitializeUndefined>");
  ReadBasicType(is, binary, &initialize_undefined);
  ExpectToken(is, binary, "<MoveSizingCommands>");
  ReadBasicType(is, binary, &move_sizing_commands);
  ExpectToken(is, binary, "<AllocateFromOther>");
  ReadBasicType(is, binary, &allocate_from_other);
  ExpectToken(is, binary, "<MinDerivTime>");
  ReadBasicType(is, binary, &min_deriv_time);
  ExpectToken(is, binary, "<MaxDerivTime>");
  ReadBasicType(is, binary, &max_deriv_time);
  if (PeekToken(is, binary) == 'M') {
    ExpectToken(is, binary, "<MaxDerivTimeRelative>");
    ReadBasicType(is, binary, &max_deriv_time_relative);
  }
  if (PeekToken(is, binary) == 'S') {
    ExpectToken(is, binary, "<SnipRowOps>");
    ReadBasicType(is, binary, &snip_row_ops);
  }
  if (PeekToken(is, binary) == 'M') {
    ExpectToken(is, binary, "<MemoryCompressionLevel>");
    ReadBasicType(is, binary, &memory_compression_level);
  }
  ExpectToken(is, binary, "</NnetOptimizeOptions>");
}

void NnetOptimizeOptions::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<NnetOptimizeOptions>");
  WriteToken(os, binary, "<Optimize>");
  WriteBasicType(os, binary, optimize);
  WriteToken(os, binary, "<ConsolidateModelUpdate>");
  WriteBasicType(os, binary, consolidate_model_update);
  WriteToken(os, binary, "<PropagateInPlace>");
  WriteBasicType(os, binary, propagate_in_place);
  WriteToken(os, binary, "<BackpropInPlace>");
  WriteBasicType(os, binary, backprop_in_place);
  WriteToken(os, binary, "<OptimizeRowOps>");
  WriteBasicType(os, binary, optimize_row_ops);
  WriteToken(os, binary, "<SplitRowOps>");
  WriteBasicType(os, binary, split_row_ops);
  WriteToken(os, binary, "<ExtendMatrices>");
  WriteBasicType(os, binary, extend_matrices);
  WriteToken(os, binary, "<ConvertAddition>");
  WriteBasicType(os, binary, convert_addition);
  WriteToken(os, binary, "<RemoveAssignments>");
  WriteBasicType(os, binary, remove_assignments);
  WriteToken(os, binary, "<AllowLeftMerge>");
  WriteBasicType(os, binary, allow_left_merge);
  WriteToken(os, binary, "<AllowRightMerge>");
  WriteBasicType(os, binary, allow_right_merge);
  WriteToken(os, binary, "<InitializeUndefined>");
  WriteBasicType(os, binary, initialize_undefined);
  WriteToken(os, binary, "<MoveSizingCommands>");
  WriteBasicType(os, binary, move_sizing_commands);
  WriteToken(os, binary, "<AllocateFromOther>");
  WriteBasicType(os, binary, allocate_from_other);
  WriteToken(os, binary, "<MinDerivTime>");
  WriteBasicType(os, binary, min_deriv_time);
  WriteToken(os, binary, "<MaxDerivTime>");
  WriteBasicType(os, binary, max_deriv_time);
  WriteToken(os, binary, "<MaxDerivTimeRelative>");
  WriteBasicType(os, binary, max_deriv_time_relative);
  WriteToken(os, binary, "<SnipRowOps>");
  WriteBasicType(os, binary, snip_row_ops);
  WriteToken(os, binary, "<MemoryCompressionLevel>");
  WriteBasicType(os, binary, memory_compression_level);
  WriteToken(os, binary, "</NnetOptimizeOptions>");
}

bool NnetOptimizeOptions::operator == (const NnetOptimizeOptions &other) const {
  return (other.optimize == optimize &&
          other.consolidate_model_update == consolidate_model_update &&
          other.propagate_in_place == propagate_in_place &&
          other.backprop_in_place == backprop_in_place &&
          other.optimize_row_ops == optimize_row_ops &&
          other.split_row_ops == split_row_ops &&
          other.convert_addition == convert_addition &&
          other.remove_assignments == remove_assignments &&
          other.allow_left_merge == allow_left_merge &&
          other.allow_right_merge == allow_right_merge &&
          other.initialize_undefined == initialize_undefined &&
          other.move_sizing_commands == move_sizing_commands &&
          other.allocate_from_other == allocate_from_other &&
          other.min_deriv_time == min_deriv_time &&
          other.max_deriv_time == max_deriv_time &&
          other.max_deriv_time_relative == max_deriv_time_relative &&
          other.snip_row_ops == snip_row_ops &&
          other.memory_compression_level == memory_compression_level);
}

// move commands that resize and zero matrices to as late/early as possible.
// (however, keep input and output commands where they were; it creates other
// headaches if we move those).
void MoveSizingCommands(const Nnet &nnet, NnetComputation *computation) {
  ComputationVariables variables;
  variables.Init(*computation);
  std::vector<CommandAttributes> attributes;
  ComputeCommandAttributes(nnet, *computation, variables, &attributes);
  std::vector<std::vector<Access> > variable_accesses;
  ComputeVariableAccesses(variables, attributes, &variable_accesses);
  std::vector<MatrixAccesses> matrix_accesses;
  ComputeMatrixAccesses(nnet, *computation, variables, attributes,
                        &matrix_accesses);

  // The way we will renumber the commands is, we will first set this vector up
  // with pairs (command-index * 3, pointer-to-command), and we will then modify
  // the command-indexes in this vector to the numbers that we want, and sort
  // it.  The reason for the * 3 is so that we can number commands "just-after"
  // existing indexes (by adding 1) and "just-before" (by subtracting 1).
  int32 num_commands = computation->commands.size(),
      num_matrices = matrix_accesses.size();

  // Matrix allocation commands tend to be followed by a command that zeroes the
  // matrix.  We want to treat the two commands as a single unit for purposes of
  // reordering.  is_command_pair[c] will be true if command c is the first
  // element of such a pair.
  std::vector<bool> is_command_pair(num_commands, false);
  for (int32 c = 0; c + 1 < num_commands; c++) {
    if (computation->commands[c].command_type == kAllocMatrix &&
        computation->commands[c+1].command_type == kSetConst &&
        computation->commands[c].arg1 == computation->commands[c+1].arg1 &&
        computation->commands[c+1].alpha == 0.0) {
      is_command_pair[c] = true;
    }
  }

  // 'command_reordering' contains (new-number, old-number) of commands.
  // the new-number is multiplied by 3 for reasons explained above.
  std::vector<std::pair<int32,int32> >
      command_reordering(num_commands);
  // Note: for now we include the second-elements-of-pairs (i.e.  the zeroing
  // commands that follow allocation commands) here; we'll ignore them later.
  for (int32 c = 0; c < num_commands; c++) {
    command_reordering[c].first = c * 3;
    command_reordering[c].second = c;
  }
  for (int32 m = 1; m < num_matrices; m++) {
    const MatrixAccesses &ma = matrix_accesses[m];
    // The following if-block relates to reordering of allocation (and,
    // implicitly, zeroing) commands.
    if (ma.allocate_command != -1 &&
        computation->commands[ma.allocate_command].command_type == kAllocMatrix) {
      // first_access_command will be index of first access, except for the
      // zeroing command that immediately follows the initialization command.
      int32 first_access_command = -1;
      // this block sets 'first_access_command'.
      if (!ma.accesses.empty()) {
        first_access_command = ma.accesses[0].command_index;
        if (first_access_command == ma.allocate_command + 1 &&
            is_command_pair[ma.allocate_command]) {
          if (ma.accesses.size() > 1)
            first_access_command = ma.accesses[1].command_index;
          else
            first_access_command = -1;
        }
      }
      if (first_access_command != -1) {
        KALDI_ASSERT(first_access_command > ma.allocate_command);
        // move the initialization command to just before the first access.
        command_reordering[ma.allocate_command].first =
            first_access_command * 3 - 1;
      }
    }
    // The following if-block relates to reordering of deallocation
    // commands.
    if (ma.deallocate_command != -1 && !ma.accesses.empty() &&
        computation->commands[ma.deallocate_command].command_type ==
        kDeallocMatrix) {
      int32 last_access_command = ma.accesses.back().command_index;
      // move the deallocation command to just after the last access.
      command_reordering[ma.deallocate_command].first =
          last_access_command * 3 + 1;
    }
  }
  std::sort(command_reordering.begin(), command_reordering.end());
  std::vector<NnetComputation::Command> reordered_commands;
  reordered_commands.reserve(num_commands);
  for (int32 c = 0; c < num_commands; c++) {
    int32 old_index = command_reordering[c].second;
    NnetComputation::Command &old_command = computation->commands[old_index];
    // the following assert is because this optimization is not allowed
    // after looped optimization.
    KALDI_ASSERT(old_command.command_type != kGotoLabel);
    if (old_index > 0 && is_command_pair[old_index - 1]) {
      // If the old command-index was a zeroing command that follows
      // an allocation command, ignore it; it will be reordered to
      // right after wherever the allocation command went, and we'll
      // deal with it when we deal with the first element of the pair.
      continue;
    } else {
      reordered_commands.push_back(computation->commands[old_index]);
      if (is_command_pair[old_index]) {
        // if this command is the first member of an (allocation, zeroing)
        // pair then we need to deal with the zeroing command as well.
        reordered_commands.push_back(computation->commands[old_index + 1]);
      }
    }
  }
  computation->commands = reordered_commands;
}

// This function removes commands of type kSetConst (with alpha=0.0), where
// possible.
void RemoveUnnecessaryZeroing(const Nnet &nnet,
                              NnetComputation *computation) {
  Analyzer a;
  a.Init(nnet, *computation);

  // OK, now we'll work out which matrices have all their pieces (i.e. all the
  // variables belonging to that matrix) written to as the first instruction
  // apart from the initial zeroing.  These matrices can have the initial
  // zeroing replaced by a sizing operation that leaves the data undefined.
  int32 num_matrices = a.matrix_accesses.size();
  for (int32 matrix_index = 0; matrix_index < num_matrices; matrix_index++) {
    const MatrixAccesses &accesses = a.matrix_accesses[matrix_index];
    if (accesses.accesses.empty())
      continue;
    int32 zeroing_command_index = accesses.accesses[0].command_index;
    NnetComputation::Command *command =
        &(computation->commands[zeroing_command_index]);
    if (!(command->command_type == kSetConst &&
          command->alpha == 0.0)) {
      continue;  // First command is not a zeroing command
    }
    // OK, the first command that accesses this matrix is a zeroing command;
    // we're going to figure out whether it was necessary.
    std::vector<int32> variables_for_matrix;
    a.variables.AppendVariablesForMatrix(matrix_index, &variables_for_matrix);
    bool all_variables_ok = true;  // if this stays true, it means we don't need
                                   // the initial zeroing.
    for (size_t i = 0; i < variables_for_matrix.size(); i++) {
      int32 variable_index = variables_for_matrix[i];
      const std::vector<Access> &v_accesses =
          a.variable_accesses[variable_index];
      if (v_accesses.size() > 1 &&
          v_accesses[1].access_type != kWriteAccess) {
        all_variables_ok = false;  // first access after zeroing was not a write
        break;
      }
      if (v_accesses.size() == 1 &&
          accesses.is_output) {
        // the only command that touches this variable is the allocation, and it
        // is an output variable.  (this is unusual, but can happen e.g. if it's
        // a derivative, but due to min_deriv_time and max_deriv_time it ends up
        // always being zero.
        all_variables_ok = false;
        break;
      }
    }
    if (all_variables_ok) {
      // Here is where the change actually happens.
      // Remove the zeroing command.
      command->command_type = kNoOperation;
    }
  }
}

/*
  This function is called from RemoveUnnecessaryAllocation.  The input is two
  sorted, unique lists, of (deallocation-commands, allocation-commands)
  e.g. (d1, d2, d3.. ), (a1, a2, a3..); and to the output is *appended* a list
  of pairs (d, a).  Each output pair must satisfy the property that d < a, and
  no member of the input lists may appear more than once in the output pairs
  (although it's OK for input a and d values not to appear in any output pairs).

  The goal of the implementation is to output as many pairs as possible, and
  secondarily for the pairs to be as close as possible to each other (to avoid
  wasting too much memory).  I'm not sure if this implementation achieves that.
*/
static void ComputeCommandPairs(
    const std::pair<std::vector<int32>, std::vector<int32> > &lists,
    std::vector<std::pair<int32,int32> > *pairs) {
  std::vector<int32> d_list = lists.first;

  std::set<int32> a_set;
  CopyVectorToSet(lists.second, &a_set);

  std::vector<int32>::reverse_iterator iter = d_list.rbegin(),
      end = d_list.rend();

  // from the latest to the earliest deallocation command...
  for (; iter != end; ++iter) {
    int32 d = *iter;
    std::set<int32>::iterator a_iter = a_set.upper_bound(d);
    // a_iter is an iterator to the first element a of the set 'a_set' such
    // that a > d, or a_set.end() if no such element exists.
    if (a_iter == a_set.end())
      continue;  // we will output no pair for this d.
    int32 a = *a_iter;
    KALDI_PARANOID_ASSERT(a > d);  // or code error
    a_set.erase(a_iter);  // remove this a from 'a_set' so it doesn't get used
                          // twice
    pairs->push_back(std::pair<int32,int32>(d, a));
  }
}

void RemoveUnnecessaryAllocation(const Nnet &nnet,
                                 NnetComputation *computation) {
  // For each size of matrix and stride-type, represented as a pair<int32,int32>
  // (the num-rows, and the num-cols * (stride-type == kDefaultStride ? 1 : -1), we
  // accumulate a list of indexes of deallocation commands that
  // are for that size, and a list of indexes of allocation commands
  // for that size.
  // For each distinct matrix size, we then call ComputeCommandPairs on those
  // two lists, to get pairs of (deallocation, allocation) command-indexes that
  // we can optimize out to a single command.

  // The map is from a (num-rows,num-columns) to two lists, of
  // (deallocation-commands, allocation-commands).  The order may seem
  // backwards, but that's the order of the pairs we are looking for.
  typedef unordered_map<std::pair<int32,int32>,
      std::pair<std::vector<int32>,std::vector<int32> >,
      PairHasher<int32> > MapType;
  MapType pair_map;
  int32 num_commands = computation->commands.size();
  for (int32 command_index = 0; command_index < num_commands; command_index++) {
    NnetComputation::Command &command = computation->commands[command_index];
    if (command.command_type == kAllocMatrix ||
        command.command_type == kDeallocMatrix) {
      int32 s = command.arg1, m = computation->submatrices[s].matrix_index,
          num_rows = computation->matrices[m].num_rows,
          num_cols = computation->matrices[m].num_cols,
          num_cols_mod = num_cols * (
              computation->matrices[m].stride_type == kDefaultStride ? 1 : -1);
      std::pair<int32,int32> p(num_rows, num_cols_mod);
      std::pair<std::vector<int32>,std::vector<int32> > &lists = pair_map[p];
      if (command.command_type == kDeallocMatrix)
        lists.first.push_back(command_index);
      else
        lists.second.push_back(command_index);
    }
  }

  MapType::const_iterator iter = pair_map.begin(), end = pair_map.end();
  std::vector<std::pair<int32,int32> > command_pairs;
  for (; iter != end; ++iter)
    ComputeCommandPairs(iter->second, &command_pairs);

  for (size_t i = 0; i < command_pairs.size(); i++) {
    int32 dealloc_index = command_pairs[i].first,
        alloc_index = command_pairs[i].second;
    NnetComputation::Command
        &dealloc_command = computation->commands[dealloc_index],
        &alloc_command = computation->commands[alloc_index];
    KALDI_ASSERT(dealloc_command.command_type ==
                 kDeallocMatrix);
    KALDI_ASSERT(alloc_command.command_type ==
                 kAllocMatrix);
    // remove the deallocation command.
    dealloc_command.command_type =  kNoOperation;
    alloc_command.arg2 = dealloc_command.arg1;
    alloc_command.command_type = kSwapMatrix;
  }
  RemoveNoOps(computation);
  FixGotoLabel(computation);
}


void VariableMergingOptimization(const NnetOptimizeOptions &config,
                                 const Nnet &nnet,
                                 NnetComputation *computation) {
  bool changed = true;
  while (changed) {
    changed = false;
    VariableMergingOptimizer opt(config, nnet, computation);
    if (opt.MergeVariables())
      changed = true;
  }
}


void ConvertAdditionToAssignment(const Nnet &nnet,
                                 NnetComputation *computation) {
  Analyzer analyzer;
  analyzer.Init(nnet, *computation);
  ComputationAnalysis analysis(*computation, analyzer);
  int32 num_commands = computation->commands.size();
  for (int32 command = 0; command < num_commands; command++) {
    NnetComputation::Command &c = computation->commands[command];
    switch (c.command_type) {
      case kMatrixAdd: case kAddRows: case kAddRowsMulti:
      case kAddToRowsMulti: {
        const std::vector<int32> &submatrices_written =
            analyzer.command_attributes[command].submatrices_written;
        KALDI_ASSERT(!submatrices_written.empty());
        std::vector<int32>::const_iterator iter = submatrices_written.begin(),
            end = submatrices_written.end();
        bool can_convert = true;
        for (; iter != end; ++iter) {
          int32 submatrix_written = *iter;
          int32 first_access_command = analysis.FirstNontrivialAccess(
              submatrix_written);
          // first_access_command is first command other than zeroing and
          // allocation that accesses this submatrix.  It can be assumed to be a
          // write command, since it makes no sense to read a variable before
          // it's written to.  If it's before this command then we need to add
          // rather than copy; we can't do the conversion to a copy command.
          if (first_access_command != command) {
            can_convert = false;
            break;
          }
        }
        if (can_convert) {  // convert to a copy command.
          switch (c.command_type) {
            case kMatrixAdd: c.command_type = kMatrixCopy;
              break;
            case kAddRows: c.command_type = kCopyRows;
               break;
            case kAddRowsMulti: c.command_type = kCopyRowsMulti;
              break;
            // note: kCopyToRowsMulti does not currently support alpha != 1.0.
            case kAddToRowsMulti: if (c.alpha == 1.0) c.command_type = kCopyToRowsMulti;
              break;
            default: KALDI_ERR << "Unexpected command type.";
          }
        }
        break;
      }
      default:
        break;
    }
  }
}


int32 MaxOutputTimeInRequest(const ComputationRequest &request) {
  int32 ans = std::numeric_limits<int32>::min();
  for (size_t i = 0; i < request.outputs.size(); i++) {
    const std::vector<Index> &indexes (request.outputs[i].indexes);
    std::vector<Index>::const_iterator iter = indexes.begin(),
        end = indexes.end();
    for (; iter != end; ++iter)
      if (iter->t > ans)
        ans = iter->t;
  }
  if (ans == std::numeric_limits<int32>::min()) {
    KALDI_ERR << "Failed to find any output indexes in computation request.";
  }
  return ans;
}


void Optimize(const NnetOptimizeOptions &config,
              const Nnet &nnet,
              int32 max_output_time_in_request,
              NnetComputation *computation) {
  if (GetVerboseLevel() >= 3) {
    CheckComputation(nnet, *computation, true);
    KALDI_LOG << "Before optimization, max memory use (bytes) = "
              << GetMaxMemoryUse(*computation);
  }

  { // Call LimitDerivativeTimes(); it's important that this
    // should come before other optimizations (search for "insist" in
    // nnet-optimize-utils.cc for the reasons).
    // this will do nothing unless --min-deriv-time or --max-deriv-time
    // or --max-deriv-time-relative was set.
    int32 max_deriv_time = config.max_deriv_time;
    if (config.max_deriv_time_relative != std::numeric_limits<int32>::max())
      max_deriv_time = config.max_deriv_time_relative +
          max_output_time_in_request;
    if (config.min_deriv_time != std::numeric_limits<int32>::min() ||
        max_deriv_time != std::numeric_limits<int32>::max())
      LimitDerivativeTimes(nnet, config.min_deriv_time,
                           max_deriv_time, computation);
  }

  if (GetVerboseLevel() >= 3)
    CheckComputation(nnet, *computation, true);

  if (config.optimize && config.consolidate_model_update) {
    ConsolidateModelUpdate(nnet, computation);

    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, true);
  }

  if (config.optimize && config.convert_addition) {
    ConvertAdditionToAssignment(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, true);
  }


  if (config.optimize &&  (config.snip_row_ops || config.optimize_row_ops ||
                           config.split_row_ops)) {
    bool must_renumber = false;
    if (config.snip_row_ops && SnipRowOps(computation))
      must_renumber = true;
    if (config.split_row_ops && SplitRowOps(computation))
      must_renumber = true;
    if (config.optimize_row_ops && ReplaceRowWithMatrixOps(computation))
      must_renumber = true;

    if (must_renumber) {
      RenumberComputation(computation);
      if (GetVerboseLevel() >= 3)
        CheckComputation(nnet, *computation, false);
    }
  }

  if (config.optimize && config.extend_matrices &&
      !config.optimize_looped_computation) {
    ExtendMatrices(computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }


  if (config.optimize &&
      (config.remove_assignments || config.backprop_in_place ||
       config.propagate_in_place)) {
    VariableMergingOptimization(config, nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  if (config.optimize && config.initialize_undefined) {
    RemoveUnnecessaryZeroing(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }


  if ((config.optimize && config.move_sizing_commands) ||
      config.optimize_looped_computation) {
    MoveSizingCommands(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  // the looped computation optimization has to go before
  // 'RemoveUnnecessaryAllocation()'.  We don't gate this by 'config.optimize'
  // because it's necessary for looped computation to run.
  if (config.optimize_looped_computation) {
    OptimizeLoopedComputation(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  if (config.optimize && config.allocate_from_other &&
      !config.optimize_looped_computation) {
    // Don't do this if it's an looped computation because we're not sure if it
    // would be correct in that case, as written.  In any case the performance
    // benefit is tiny.
    RemoveUnnecessaryAllocation(nnet, computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  // The following is not configurable because it is necessary for
  // the computation to run correctly (we do it after compilation too,
  // but the operations may have been put out of order by
  // other optimizations.)
  ConsolidateIoOperations(nnet, computation);

  if (config.optimize_looped_computation)
    FixGotoLabel(computation);


  if (config.memory_compression_level > 0 &&
      !config.optimize_looped_computation) {
    OptimizeMemoryCompression(nnet, config.memory_compression_level,
                              computation);
    if (GetVerboseLevel() >= 3)
      CheckComputation(nnet, *computation, false);
  }

  if (GetVerboseLevel() >= 3) {
    CheckComputation(nnet, *computation, false);
    KALDI_LOG << "After optimization, max memory use (bytes) = "
              << GetMaxMemoryUse(*computation);
  }
}


CachingOptimizingCompiler::CachingOptimizingCompiler(
    const Nnet &nnet,
    const CachingOptimizingCompilerOptions config):
    nnet_(nnet), config_(config),
    seconds_taken_total_(0.0), seconds_taken_compile_(0.0),
    seconds_taken_optimize_(0.0), seconds_taken_expand_(0.0),
    seconds_taken_check_(0.0), seconds_taken_indexes_(0.0),
    seconds_taken_io_(0.0), cache_(config.cache_capacity) { }

CachingOptimizingCompiler::CachingOptimizingCompiler(
    const Nnet &nnet,
    const NnetOptimizeOptions &opt_config,
    const CachingOptimizingCompilerOptions config):
    nnet_(nnet), config_(config), opt_config_(opt_config),
    seconds_taken_total_(0.0), seconds_taken_compile_(0.0),
    seconds_taken_optimize_(0.0), seconds_taken_expand_(0.0),
    seconds_taken_check_(0.0), seconds_taken_indexes_(0.0),
    seconds_taken_io_(0.0), cache_(config.cache_capacity) { }


void CachingOptimizingCompiler::ReadCache(std::istream &is, bool binary) {
  {
    Timer timer;
    NnetOptimizeOptions opt_config_cached;
    opt_config_cached.Read(is, binary);
    // we won't read cached computations if any optimize option has been changed.
    if (!(opt_config_ == opt_config_cached))
      return;
    cache_.Read(is, binary);
    seconds_taken_io_ += timer.Elapsed();
  }
  if (GetVerboseLevel() >= 2) {
    Timer timer;
    cache_.Check(nnet_);
    seconds_taken_check_ += timer.Elapsed();
    // we consider the check time part of the total time...  this is very
    // arbitrary but it only affects printed times-taken.
    seconds_taken_total_ += timer.Elapsed();
  }

}

void CachingOptimizingCompiler::WriteCache(std::ostream &os, bool binary) {
  Timer timer;
  opt_config_.Write(os, binary);
  cache_.Write(os, binary);
  seconds_taken_io_ += timer.Elapsed();
}

CachingOptimizingCompiler::~CachingOptimizingCompiler() {
  if (seconds_taken_total_ > 0.0 || seconds_taken_io_ > 0.0) {
    std::ostringstream os;
    double seconds_taken_misc = seconds_taken_total_ - seconds_taken_compile_
        - seconds_taken_optimize_ - seconds_taken_expand_
        - seconds_taken_check_ - seconds_taken_indexes_;
    os << std::setprecision(3) << seconds_taken_total_
       << " seconds taken in nnet3 compilation total (breakdown: "
       << seconds_taken_compile_ << " compilation, "
       << seconds_taken_optimize_ << " optimization, "
       << seconds_taken_expand_ << " shortcut expansion, "
       << seconds_taken_check_ << " checking, "
       << seconds_taken_indexes_ << " computing indexes, "
       << seconds_taken_misc << " misc.) + "
       << seconds_taken_io_ << " I/O.";
    KALDI_LOG << os.str();
    // note: the leftover amount is misc things like hashing and == comparisons on
    // computation-requests, and calling RequestIsDecomposable().
  }
}

std::shared_ptr<const NnetComputation> CachingOptimizingCompiler::Compile(
    const ComputationRequest  &in_request) {
  Timer timer;
  std::shared_ptr<const NnetComputation>  ans = CompileInternal(in_request);
  seconds_taken_total_ += timer.Elapsed();
  return ans;
}

std::shared_ptr<const NnetComputation> CachingOptimizingCompiler::CompileInternal(
    const ComputationRequest  &request) {
  std::shared_ptr<const NnetComputation> ans = cache_.Find(request);
  if (ans != NULL) {
    return ans;
  } else {
    const NnetComputation *computation = NULL;
    if (config_.use_shortcut)
      computation = CompileViaShortcut(request);
    if (computation == NULL)
      computation = CompileNoShortcut(request);
    KALDI_ASSERT(computation != NULL);
    return cache_.Insert(request, computation);
  }
}


const NnetComputation *CachingOptimizingCompiler::CompileNoShortcut(
    const ComputationRequest &request) {

  Compiler compiler(request, nnet_);
  // note: 'opts' only contains 'output_debug_info', which is true by default.
  // There may be situations where we'd prefer not to keep it, for speed.
  CompilerOptions opts;
  NnetComputation *computation = new NnetComputation;

  {
    Timer timer;
    compiler.CreateComputation(opts, computation);
    seconds_taken_compile_ += timer.Elapsed();
  }

  int32 verbose_cutoff = 4;
  if (GetVerboseLevel() >= verbose_cutoff) {
    std::ostringstream os1;
    request.Print(os1);
    KALDI_LOG << "Computation request is " << os1.str();
    std::ostringstream os2;
    computation->Print(os2, nnet_);
    KALDI_LOG << "Generated computation is: " << os2.str();
  }

  { // some checking.  Note: there may come a time when we might
    // prefer to disable this checking.
    Timer timer;
    CheckComputationOptions check_config;
    // we can do the rewrite check since it's before optimization.
    check_config.check_rewrite = true;
    ComputationChecker checker(check_config, nnet_, *computation);
    checker.Check();
    seconds_taken_check_ += timer.Elapsed();
  }

  {
    Timer timer;
    Optimize(opt_config_, nnet_,
             MaxOutputTimeInRequest(request),
             computation);
    seconds_taken_optimize_ += timer.Elapsed();
  }

  if (GetVerboseLevel() >= verbose_cutoff) {
    std::ostringstream os;
    computation->Print(os, nnet_);
    KALDI_LOG << "Optimized computation is: " << os.str();
  }

  {  // check the computation again.
    Timer timer;
    CheckComputationOptions check_config;
    ComputationChecker checker(check_config, nnet_, *computation);
    checker.Check();
    seconds_taken_check_ += timer.Elapsed();
  }

  {
    Timer timer;
    computation->ComputeCudaIndexes();
    seconds_taken_indexes_ += timer.Elapsed();
  }
  return computation;
}


const NnetComputation *CachingOptimizingCompiler::CompileViaShortcut(
    const ComputationRequest &request) {
  int32 num_n_values;
  ComputationRequest mini_request;
  if (!RequestIsDecomposable(request, &mini_request, &num_n_values))
    return NULL;

  // By invoking CompileInternal() on the mini request, we go through the same
  // caching process as for any externally requested computation.
  std::shared_ptr<const NnetComputation> mini_computation =
      CompileInternal(mini_request);

  // note: by default we always create debug_info, even in regular compilation.
  // (e.g. it defaults to true in CompilerOptions).  If it really seems to be a
  // significant overhead, we can revisit this at some point in future.
  bool need_debug_info = true;


  NnetComputation *ans = new NnetComputation();

  {
    Timer timer;
    ExpandComputation(nnet_, request.misc_info, *mini_computation,
                      need_debug_info, num_n_values, ans);
    seconds_taken_expand_ += timer.Elapsed();
  }
  if (GetVerboseLevel() >= 3) {
    CheckComputation(nnet_, *ans, false);
  }

  {
    Timer timer;
    ans->ComputeCudaIndexes();
    seconds_taken_indexes_ += timer.Elapsed();
  }
  return ans;
}



/// Split the computation up into segments bounded by kNoOperationMarker.  For
/// each segment, a pair of command-indexes (start, end) is output to the vector
/// 'segments', so the commands in the segment (not including
/// kNoOperationMarker) are numbered from start ... end - 1.
static void SplitComputationIntoSegments(
    const NnetComputation &computation,
    std::vector<std::pair<int32, int32> > *segments) {

  int32 num_commands = computation.commands.size();
  segments->clear();
  int32 cur_start = 0;
  for (int32 c = 0; c < num_commands; c++) {
    if (computation.commands[c].command_type == kNoOperationMarker) {
      segments->push_back(std::pair<int32, int32>(cur_start, c));
      cur_start = c + 1;
    }
  }
  segments->push_back(std::pair<int32, int32>(cur_start, num_commands));
}


void ConsolidateIoOperations(const Nnet &nnet,
                             NnetComputation *computation) {
  // These segments, represented as (start-index, end-index),
  // are segments of the computation separated by kNoOperationMarker.
  std::vector<std::pair<int32, int32> > segments;
  SplitComputationIntoSegments(*computation, &segments);

  int32 num_commands = computation->commands.size();
  std::vector<NnetComputation::Command> reordered_commands(num_commands);
  // put kNoOperationMarker between all segments in the reordered commands.
  for (size_t s = 0; s + 1 < segments.size(); s++)
    reordered_commands[segments[s].second].command_type = kNoOperationMarker;

  // for each segment we'll divide the commands up into those that must appear
  // at the left of the segment (kAcceptInput for inputs and output-derivs), those
  // that must appear in the middle (most commands), those that must appear
  // on the right (kProvideOutput for output nodes and input derivatives).
  std::vector<int32> left_commands, middle_commands, right_commands;

  for (size_t s = 0; s < segments.size(); s++) {
    int32 segment_start = segments[s].first,
        segment_end = segments[s].second;
    left_commands.clear();
    middle_commands.clear();
    right_commands.clear();
    for (int32 c = segment_start; c < segment_end; c++) {
      if (computation->commands[c].command_type == kProvideOutput) {
        right_commands.push_back(c);
      } else if (computation->commands[c].command_type == kAcceptInput) {
        left_commands.push_back(c);
      } else {
        middle_commands.push_back(c);
      }
    }
    std::vector<int32>::const_iterator iter = left_commands.begin(),
        end = left_commands.end();
    int32 c = segment_start;
    for (; iter != end; ++iter, ++c)
      reordered_commands[c] = computation->commands[*iter];
    iter = middle_commands.begin();
    end = middle_commands.end();
    for (; iter != end; ++iter, ++c)
      reordered_commands[c] = computation->commands[*iter];
    iter = right_commands.begin();
    end = right_commands.end();
    for (; iter != end; ++iter, ++c)
      reordered_commands[c] = computation->commands[*iter];
    KALDI_ASSERT(c == segment_end);
  }
  computation->commands.swap(reordered_commands);
}




} // namespace nnet3
} // namespace kaldi
