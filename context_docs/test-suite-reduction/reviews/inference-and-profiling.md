# Report: tests/unit/inference + profiling + inference_stack/deadline

## tests/unit/inference/test_scheduling.py — TRIM — tests 138→106, LOC 2957→~2570, cpu-s 3.0→2.5
- TestJoinedGpuCount::test_no_gpu_nodes_yet_reads_zero — PARAM — dict.get default; test_reads_the_cluster_total + cpu-only case survive
- TestPollTracker::test_stall_log_level[inference|loading] — PEDANTIC — log level only; recovery/abort tests cover the stall decision
- TestPollTracker::test_systemic_stall_raises[4] — PARAM — [3] is the boundary; above-boundary adds nothing
- TestPollTracker::test_tracker_error_swallowed — DUP — test_a_failed_poll_recovers_nothing makes the same call and also asserts the return
- TestPollTracker::test_the_line_names_only_cells_this_snapshot_counted[one-cell,unnamed] — PARAM — chained-boundary and not-yet-reporting carry the logic
- TestPollTracker::test_phase_summary_logged — PEDANTIC — asserts words "inference"/"loading"/"active" in a log line
- TestActorPoolProperties::test_busy_actors_property — PEDANTIC — property accessor; exercised by every retire/dispatch test
- TestActorPoolProperties::test_live_count_decrements_on_retire — PEDANTIC — set arithmetic; TestRetireIdle asserts live_count throughout
- TestResolveIid::test_no_op_when_no_pending_ref — PEDANTIC — no assertion
- TestResolveIid::test_resolves_when_ref_ready — DUP — TestResolveInitializing::test_moves_resolved_actor_out_of_initializing
- TestResolveIid::test_non_blocking_leaves_ref_pending — DUP — TestResolveInitializing::test_leaves_unresolved_actors_in_initializing
- TestReservations::test_take_reserved_pops — PEDANTIC — dict.pop wrapper
- TestReservations::test_no_queue_means_no_reservation — PARAM — test_shallow_queue_does_not_reserve covers the no-hint branch
- TestRetireIdle::test_grace_period_always_applies — DUP — test_grace_period_not_elapsed_no_kill
- TestRetireIdle::test_floor_check_prevents_kill — DUP — test_outstanding_work_still_outranks_a_smaller_floor (same work-floor rule)
- TestRetireIdle::test_on_retire_not_called_when_none — PEDANTIC — None default is exercised by every test without a callback
- TestMarkInitializing::test_sets_up_state — DUP — TestAddActors::test_appends_and_marks_initializing asserts the same state
- TestMarkInitializing::test_custom_placeholder — DUP — TestReplace::test_replace_creates_new_actor_and_marks_initializing asserts the replacement placeholder
- TestAddActors::test_added_actor_gets_work_via_dispatch_idle — DUP — TestResolveInitializing::test_newly_ready_actor_receives_work_via_dispatch_idle (same flow)
- TestReplace::test_death_log_level[1-4,2-4,4-4] — PEDANTIC — log level escalation only; removes helper _do_replace
- TestBatchActorsToRequest::test_final_batch_clamped_to_target — DUP — TestActorRequestHeadroom::test_the_final_batch_is_clamped_to_the_target's headroom=None control
- TestActorRequestHeadroom::test_readiness_is_not_what_counts — DUP — identical call to first assert of test_each_placement_earns_the_right_to_ask_for_more
- TestWorkStealingGpuHours::test_a_failed_capacity_lookup_does_not_abort_the_run — DUP — TestJoinedGpuCount::test_a_failed_lookup_carries_the_previous_count_forward (the guard lives there)
- TestWorkStealingBatching::test_no_batching_when_factory_absent — HISTORY — its guard patches ray.nodes, which scheduling.py no longer calls (grep: 0 hits); vacuous
- TestChainedWorkSource::test_retirement_runs_once_the_source_is_exhausted — DUP — TestRetireIdleGate::test_default_retires_idle_actors
- TestChunkUid (all 3) — PEDANTIC/DUP — string format of chunk_uid; distinct-uid guarantee asserted in TestChainedWorkSource::test_zones_stream_with_their_own_contexts
Notes: fast (mocked Ray); ~2.9 s is spread thin. `ray.nodes` patches in test_second_batch_requested_after_first_placed and TestMidRunWindDownNeedsRegrowth._floors are dead scaffolding (src never calls ray.nodes): delete the patch lines. Six loop tests rebuild the same `config = MagicMock(); checkpoint_path=...` plus a fake_wait/fake_get pair: a `_mock_config(**attrs)` helper and one `_ordered_wait(order)` helper would remove ~120 LOC. Merge test_replace_carries_credentials_and_region_to_new_actor into TestReplace::test_replace_creates_new_actor_and_marks_initializing (same setup). Both headroom=None (legacy default, config default None) and headroom paths are live; keep both suites.

## tests/unit/inference/test_gpu_starvation.py — DELETE (relocate) — tests 29→0, LOC 1482→0, cpu-s 0.0→0.0
- whole file — HISTORY — a campaign-planning MODEL, not a test: imports nothing from src except, via tests/unit/zone_density.plan, `_partition_by_live_tiles`; every assertion checks arithmetic defined in this same file (RadarCell/OrbitBand tables, ingest_hours, _landings, CAMPAIGN_* outturn constants)
Notes: Move it verbatim to scripts/planning/gpu_starvation.py (asserts become a `main()` self-check, `test_report`/`test_cost_report` become its printed output) or to context_docs/campaign/ beside campaign-cost-model.md; it is the campaign's inference cost record, so keep the content, just out of `pytest`. Update path references in context_docs/campaign/campaign-plan.md:336, context_docs/corrections-register.md:75, campaign-cost-model.md (2 refs), ADR 015:89. tests/unit/zone_density.py stays (tests/unit/orchestration/flows/test_cluster_balance.py, another slice, still uses it). Saves no time, but 1,482 LOC and 29 "tests" that cannot fail from any src change except the partitioner, which test_cluster_balance already covers.

## tests/unit/inference/test_strip_loop.py — TRIM — tests 48→39, LOC 1231→~1000, cpu-s 7.2→~5.0
- TestStripHeightForDensity::test_single_strip_when_full_height_fits_one_budget — DUP — TestStripPlan::test_fits_one_budget_is_single_strip (same t_kept=30, 2000²)
- TestStripPlan::test_dense_chunk_splits_with_every_set_under_one_budget — DUP — TestStripHeightForDensity::test_dense_single_chunk_splits_to_respect_budget (_strip_plan just slices that height)
- TestProcessChunkStriping::test_multi_strip_actually_splits — PEDANTIC — asserts a fixture constant (_CHUNK.height > 4)
- TestDeferredStagingWrites::test_flush_with_nothing_pending_returns_none — PEDANTIC — the None branch is exercised by every _run_* helper's `flushed is None or ...`
- TestEastingBboxCrop::test_crop_reads_only_valid_columns — DUP — test_cropped_outputs_bit_identical_to_uncropped asserts the cropped x_slice (tighten it to the exact slice 2:5)
- TestXChunkPrefetch::test_prefetched_chunk_is_bit_identical_to_serial[one-strip] — PARAM — multi-strip survives; it covers the stash's first-strip handoff that one-strip trivially does
- TestPerCellTimeWindow::test_passing_the_config_window_explicitly_is_bit_identical — HISTORY — default resolves to one local (src); default path is every other test, override path is test_a_different_window_actually_changes_what_is_read
- TestPerCellTimeWindow::test_the_zone_context_carries_the_window_to_the_dispatch — PEDANTIC — frozen-dataclass defaults/equality; inline import
- TestTheRecordTheConsumerWillRead::test_the_record_carries_the_grid_it_was_produced_on — DUP — fold its one assert into test_the_consumer_grades_the_producers_record_as_consistent (same uncropped run)
Notes: SLOW because ~35 tests each call the real `InferenceActor.process_chunk` (CPU model from function-scoped `test_model`, writer + prefetch thread pools spun per actor), and the bit-identity tests run it 2-3 times. Cuts above remove ~8 process_chunk runs. Next cheapest fix: a module-scoped fixture caching the serial reference write of `_CHUNK` under `_force_strip_plan(4)` (reused by test_single_vs_multi_strip_identical, test_a_holes_tile..., TestXChunkPrefetch); `inference_config` is mutated per test so copy it (`dataclasses.replace`) rather than widen its scope. LOC: five near-identical synthetic-S2-store builders (all_empty, _make_half_empty_stores, _make_sliver_stores, _stores_with_sar_gap, _refused_run) and five `_run` wrappers collapse into one `_s2_group(scl, n_t, start)` + one `_run_process_chunk(opener=, patches=, **cfg)` (~150 LOC).

## tests/unit/inference/test_data_loading.py — TRIM — tests 43→32, LOC 593→~470, cpu-s 3.9→~1.5
- TestComputeDoy::test_doy_values — DUP — tests/unit/storage/test_zarr_store.py (~line 457) asserts the same 2024 Jan 1 / Jun 15 DOYs; compute_doy lives in storage/
- TestComputeDoy::test_non_leap_year — DUP — fold the one 2023-12-31→365 assert into that storage test
- TestLoadSarOrbit::test_the_months_returned_line_up_with_the_doys_they_came_from — PEDANTIC — asserts only shape equality, already implied by test_shapes_and_dtypes
- TestLoadSarOrbit::test_store_path_uses_orbit_name — DUP — TestLoadChunkOrchestration ascending/descending tests assert the sar_<orbit> paths opened
- TestLoadS2::test_obs_count_from_full_mask — PEDANTIC — `max <= n_t` holds for any count; real count checked by TestSharedMaskBundle and test_strip_loop month-coverage test
- TestLoadChunkOrchestration::test_descending_only_skips_ascending — PARAM — mirror of test_ascending_only_skips_descending; parametrize one test over orbit
- TestLoadChunkOrchestration::test_obs_counts_populated — PEDANTIC — not-None + shape; test_empty_orbit_obs_count_is_zero and strip reassembly check values
- TestStripReassembly::test_y_sub_none_matches_full_chunk_dims — PEDANTIC — dims of an unsliced load
- TestStripReassembly::test_strip_dims_reflect_strip — DUP — test_strips_reassemble_to_whole_chunk fails on any wrong strip shape
- TestStripReassembly::test_strip_reads_identical_pixels — DUP — test_strip_loop.py::TestProcessChunkStriping::test_single_vs_multi_strip_identical covers SAR through the actor (bit-identical embeddings + S1 counts)
- TestResolveS1Orbit::test_both_with_only_descending_downgrades — PARAM — keep only_ascending (or parametrize the two)
Notes: ~3 of the 3.9 cpu-s is TestLoadS2Bands::test_concurrent_band_read_matches_serial_on_real_icechunk: a real local icechunk write_dataset of 8×128×128×10 bands, then 10 repeated threaded reads. Keep it (only test against icechunk's Rust backend) but cut to 64×64 with 32-px spatial chunks and 3 repeats; a deterministic race this small will not surface in 10 tries that it would not in 3. Its inline import of zarr_store should move to the top. Three classes redefine an identical 12-row `_CHUNK` and `_side_effect`; one module-level `_TALL_CHUNK` + `_opener()` removes ~25 LOC.

## tests/unit/inference/test_models.py — TRIM — tests 32→23, LOC 389→~290, cpu-s 0.8→0.6
- TestTransformerEncoder::test_s2_output_shape — DUP — TestBuildInferenceModel::test_build_from_config forwards both backbones and asserts output shape
- TestTransformerEncoder::test_s1_output_shape — DUP — same
- TestTemporalPositionalEncoder::test_output_shape — DUP — every TestPositionalEncoderBitIdentity assert_close checks shape and values
- TestCustomGRU::test_cell_output_shape — DUP — test_tessera_gru_formulation assert_close checks shape
- TestCustomGRU::test_gru_output_shape — DUP — TestTemporalAwarePooling::test_matches_the_trained_formula runs CustomGRU via _reference_pool
- TestMultimodalBTInferenceModel::test_forward_shape_concat — DUP — TestBuildInferenceModel::test_build_from_config (concat is the test config's fusion)
- TestLoadCheckpoint::test_preserves_tensor_values — PEDANTIC — dict-key rename cannot alter tensors; test_strips_orig_mod_prefix covers the rename
- TestPositionalEncoderBitIdentity::test_float32_matches_within_one_ulp — DUP — test_every_day_of_year_matches_the_per_pixel_encoding[float32] covers all 367 days
- TestPositionalEncoderBitIdentity::test_doy_above_256_keeps_one_day_resolution — DUP — the [bfloat16] case asserts exact match on days 257/258 against the FP32 reference
Notes: the 6 TestLoadCheckpoint tests each inline-import load_v11_checkpoint; hoist to the top. GPU-only tests skip on CPU and cost nothing; keep (they run on GPU hosts).

## tests/unit/inference/test_inference_loop.py — TRIM — tests 12→6, LOC 329→~190, cpu-s 2.4→~1.3
- TestRunInference::test_output_shape — PEDANTIC — shapes/dtypes; test_pipelined_output_matches_serial_reference compares full arrays against a typed reference
- TestRunInference::test_nonzero_embeddings_for_valid_pixels — DUP — test_partial_validity_marks_only_invalid_pixels (finite scales on valid pixels) and the pipelined equality test
- TestPrepareGpu::test_cpu_no_side_effects — DUP — test_cpu_leaves_the_model_in_float32 (same call, same guarantee)
- TestProcessSubBatch::test_transfer_and_forward_returns_quantized_host_rows — PEDANTIC — shapes, dtypes, `secs >= 0`; primitive is driven by the pipelined-vs-serial test
- TestProcessSubBatch::test_write_quantized_rows_writes_to_flat_out — DUP — scatter correctness asserted by test_partial_validity_marks_only_invalid_pixels / test_invalid_pixels_are_zero through run_inference
- TestThroughputAccounting::test_final_log_reports_tokens_and_tflops — PEDANTIC — asserts two substrings in a log line
Notes: each test builds a dataset and runs run_inference on CPU (~0.2 s each). Inline imports in test_pipelined_output_matches_serial_reference (duplicates the module-level import), TestThroughputAccounting and TestPipelinedGpuLoop should be hoisted. GPU test skips on CPU by documented decision; keep.

## tests/unit/inference/test_sampling_v2_parity.py — TRIM — tests 7→4, LOC 150→~95, cpu-s 2.4→~0.9
- test_v11_is_untouched_by_the_addition — DEAD — exercises resampler_for (sampling.py:95), which no src/scripts caller reaches (dataset dispatches via resample_s2/s1_bucket); 8,451-pair loop
- test_an_unknown_model_version_is_refused_rather_than_defaulted — DEAD — same function; delete resampler_for from src (and its inference/README.md:516 mention)
- test_the_two_rules_really_do_disagree — DUP — test_the_dataset_resamples_with_the_version_it_was_given asserts v1.1 vs v2 sequences differ through the real dataset; second 8,451-pair loop
- (trim, not removal) test_the_actor_passes_its_configured_version_to_the_dataset — PEDANTIC — greps actor source text for a literal kwarg; delete it too if the owner accepts source-text tests as pedantry (tests 4→3)
Notes: time is the three exhaustive (count, bucket) loops over DEFAULT_NUM_OBS_CHECKPOINTS; keeping only the upstream-parity loop removes two-thirds of it. In test_normalisation_and_resampling_cannot_describe_different_students drop the `inspect.signature` asserts (HISTORY: the parameter's absence is structural) and keep the band-mean value check; hoist its three inline imports.

## tests/unit/inference/test_student_v2.py — TRIM — tests 31→22, LOC 319→~240, cpu-s 1.2→1.0
- test_attention_pooling_shape — DUP — test_attention_pooling_empty_sequence_is_zeros and _is_a_convex_combination exercise the same forward
- test_v2_backbone_has_no_recurrence — DUP — test_v2_model_state_dict_keys_match_checkpoint_layout asserts no temporal_context/GRU keys
- test_v2_backbone_output_shape[10-11,2-3] — DUP — test_v2_runs_the_unmodified_inference_loop forwards both backbones to the (H,W,128) output
- test_v2_dim_reducer_shapes_and_keys — DUP — test_v2_model_parameter_count + state-dict key test pin the reducer layout
- test_v2_model_output_shape_is_128 — DUP — test_v2_runs_the_unmodified_inference_loop asserts (4,4,128)
- test_quantization_is_version_agnostic — DUP — CPU/torch quantize parity and error bound are test_quantization.py's job; v2 rows reach it in test_v2_runs_the_unmodified_inference_loop
- test_v2_arch_spec_is_registered — PEDANTIC — asserts two registry constants
- test_qk_norm_layer_preserves_shape — IMPROBABLE — QKNormEncoderLayer is unreachable: the only registered family sets enable_qk_norm=False (config/inference.py:471); keep only if a QK-norm checkpoint is planned
Notes: `v2_model` is already module-scoped (43.8M-param build). Hoist the inline imports in test_quantization_is_version_agnostic and test_v2_runs_the_unmodified_inference_loop. test_qk_norm_layer_rejects_indivisible_head_dim goes with QKNormEncoderLayer if the class is removed.

## tests/unit/inference/test_sampling_v11.py — TRIM — tests 19→9, LOC 308→~190, cpu-s 0.0→0.0
- test_bucket_for_count_exact_match / _between / _above_max_clamps / _zero_returns_first / _negative_returns_first (5) — DEAD — bucket_for_count (sampling.py:152) has no src/scripts caller; compute_bin_keys is the live path (test_compute_bin_keys_values covers exact/between/clamp/zero). Delete the function too.
- test_compute_bin_keys_shape_and_dtype — DUP — test_compute_bin_keys_values indexes keys["s2"]/["s1"]
- test_resample_s2_bucket_output_shape — DUP — TestVectorisedS2MatchesLoop::test_bit_identical_across_targets compares whole arrays
- test_resample_s2_bucket_no_valid_pixels_returns_zeros — DUP — TestVectorisedS2MatchesLoop::test_all_pixels_dead
- test_resample_s1_bucket_both_orbits_output_shape — DUP — TestVectorisedS1MatchesLoop::test_bit_identical_both_orbits
- test_resample_s1_bucket_single_orbit_empty_desc — DUP — the (3, 0, 8) case of test_bit_identical_both_orbits

## tests/unit/inference/test_dataset_v11.py — TRIM — tests 20→16, LOC 311→~270, cpu-s 0.0→0.0
- test_bucket_assignment_valid_pixels — PEDANTIC — asserts `len(ds) > 0` and len == sum of an internal dict
- test_iter_buckets_keys_are_valid_checkpoints — DUP — test_get_bucket_batch_shapes and the allow_s2_only test assert bucket keys/sizes
- test_bucket_sizes_consistency — PEDANTIC — accessor vs the private dict it reads
- test_s2_only_pixels_dropped_by_default — DUP — TestTheMinimumOpticalDepthGate::test_a_deep_pixel_with_no_radar_is_embedded... asserts the allow_s2_only=False contrast (0 embedded, all refused_no_radar)
Notes: test_iter_buckets_largest_first is guarded by `if len(bucket_list) > 1` and can pass asserting nothing; drop the guard and assert >1 bucket. `_make_chunk_data`/CKPS are imported by test_sampling_v2_parity.py, so keep them here (or move to a shared helper).

## tests/unit/inference/test_shard_record.py — DELETE — tests 18→0, LOC 302→0, cpu-s 0.0→0.0
- whole file — DEAD — `grep -rn shard_record src scripts` finds no importer outside the module itself; ADR 040:145 already lists inference/shard_record.py as "used only by tests". Unwired since 2026-08-13 (commits f2b018e3, 2f1a6cdd). The live per-shard record is actors._coverage_record (tested in test_strip_loop.py::TestTheRecordCountsOnlyTheEvaluatedColumns).
Notes: delete src/tessera_embeddings/inference/shard_record.py with it, unless the owner confirms a pending branch wires it in (then keep only TestTheDenominator, TestTheTwoImplementationsAgree and TestTheChunkBitmask::test_the_top_bit_survives_the_row_encoding).

## tests/unit/inference/test_resource_monitor.py — TRIM — tests 20→18, LOC 310→~270, cpu-s 0.0→0.0
- TestResourceMonitorLine::test_no_index_means_no_stamp — PEDANTIC — absence of a substring in a log line
- TestHostRamPeak::test_the_peak_is_on_the_line — PEDANTIC — exact log-format string "RAMpeak=19.2/32.0 GB (60%)"; peak value already pinned by test_the_peak_survives_a_later_lower_sample
Notes: the two TestResourceMonitorLine tests repeat a 10-line `_get_gpu_stats` return dict; one module constant `_STATS` removes ~15 LOC.

## tests/unit/inference/test_inference_actor.py — TRIM — tests 14→9, LOC 269→~200, cpu-s 0.1→0.1
- test_actor_options_accepts_cpu_and_gpu — PEDANTIC — asserts Ray's `.options()` returns something callable; Ray's API, not ours
- test_actor_is_a_ray_remote_class — PEDANTIC — isinstance on the decorator's product
- test_download_checkpoint_leaves_no_partial_files — DUP — test_download_checkpoint_concurrent_callers_get_intact_file asserts the same single-file directory under contention
- test_the_actor_hands_its_own_gpu_index_to_the_monitor — PEDANTIC — greps actors.py source for a literal line; breaks on reformatting, passes on a dead copy of the line
- test_the_ram_peak_is_reset_at_each_chunk_not_once_per_actor — PEDANTIC — same source-text grep pattern
Notes: if the owner wants the two wiring guarantees kept, replace both greps with one behavioural test on the bare actor from test_strip_loop's `_make_actor` (assert `_resource_monitor.peak_host_ram_gib()` is None after process_chunk's prologue). Hoist the inline imports in test_configure_actor_logging_enables_debug_on_real_module_loggers.

## tests/unit/inference/test_student_v2_golden.py — KEEP (skip-by-default earns its LOC) — tests 7→5, LOC 223→~200, cpu-s 0.0→0.0
- test_state_dict_keys_are_identical — DUP — test_weights_loaded_are_identical iterates every key and would KeyError on a missing one; strict=True load rejects extras
- test_upstream_output_is_layer_normalised — PEDANTIC — a check on the vendored reference, not on our port; test_student_v2.py::test_v2_output_is_layer_normalised_per_row covers ours
Notes: costs nothing by default and is the ONLY bit-level evidence the v2 port matches upstream on real weights; the sha256 gate before upstream's weights_only=False unpickle is a genuine safety guard. Keep, run by hand when student_v2.py changes.

## tests/unit/inference/test_v11_golden.py — KEEP (skip-by-default earns its LOC) — tests 5→5, LOC 144→144, cpu-s 0.0→0.0
Notes: the only test that would have caught the ADR 026 nn.GRU swap; zero default cost. Its checkpoint-digest fixture, by-path upstream import and `_checkpoint_path()` duplicate test_student_v2_golden.py's; merging both into one `test_upstream_golden.py` parametrised over (env var, sha256, size, reference loader) saves ~70 LOC — optional.

## tests/unit/inference/test_output_store_preflight.py — TRIM — tests 9→7, LOC 219→~185, cpu-s 0.2→0.15
- test_matching_store_passes — DUP — test_the_matching_family_still_passes_the_gate builds the same store (plus the attr) and calls the gate with the same config
- test_the_output_path_has_one_definition — PEDANTIC — string format of embedding_store_path
Notes: test_a_cross_family_append_is_refused_before_any_compute and test_a_legacy_store_with_no_encoder_attr_is_read_as_v11 build the identical v2_config inline; parametrise them over encoder_url_value (encoder_url(DEFAULT), None), ~25 LOC.

## tests/unit/inference/test_quantization.py — TRIM — tests 12→8, LOC 191→~145, cpu-s 0.0→0.0
- test_quantize_embeddings_delegates_consistently — DEAD — quantize_embeddings has no src/scripts caller (grep: only inference/README.md:716 prose); ADR 040:145 lists it test-only. Delete it and the README sentence.
- test_round_trip_recovers_within_scale — DEAD — goes through quantize_embeddings + dequantize_embeddings, both test-only; the error bound follows from test_quantize_rows_matches_whole_array_reference
- test_constant_row_saturates_to_127 — DUP — test_clip_bound_is_symmetric_127 pins the ±127 saturation
- test_raise_on_nonfinite_scales_accepts_finite — PEDANTIC — no assertion; the finite path runs in every inference test

## tests/unit/inference/test_chunk_spec.py — TRIM — tests 15→10, LOC 134→~100, cpu-s 0.1→0.1
- TestChunkSpec::test_frozen — PEDANTIC — dataclass(frozen=True) behaviour
- TestChunkSpec::test_edge_chunk_smaller — DUP — TestEnumerateChunks::test_edge_chunks asserts edge heights/widths via enumerate_chunks
- TestEnumerateChunks::test_single_chunk — DUP — parametrized count case and test_edge_chunks
- TestEnumerateChunks::test_parametrized_counts[1500-1500-1500-1, 1-1-1500-1] — PARAM — 1501 and 10000 cases carry the ceil logic; test_exact_fit covers the even case

## tests/unit/inference/test_assessed_window_coverage.py — MERGE INTO tests/unit/inference/test_data_loading.py — tests 8→8, LOC 74→~55, cpu-s 0→0 (same module under test, `_months_within_assessed`; the five boundary tests collapse into one parametrised table)
## tests/unit/inference/test_orchestration_helpers.py — TRIM — tests 3→3, LOC 52→~30, cpu-s 0→0 (three copies of one test: parametrise over orbit → expected keys)
## tests/unit/inference/test_runner_resume.py — TRIM — tests 4→3, LOC 46→~38, cpu-s 0→0
- test_the_two_outcomes_are_distinguishable — DUP — test_a_restored_skip_reports_as_skipped + test_a_restored_staged_tile_reports_as_success already pin the two distinct statuses
## tests/unit/inference_stack.py — KEEP — shared constants imported by 10+ assembly/runner/flow test files
## tests/unit/deadline.py — KEEP — 31-line hang-to-failure helper shared by watchdog and drain tests

## tests/unit/profiling/test_profiling_tools.py — TRIM — tests 23→15, LOC 359→~270, cpu-s 0.7→0.5
- TestResourceMonitorContext::test_clearing_a_slot_removes_it — PEDANTIC — absence of "work:" in a log line; the slot dict's None-removal is one line
- TestChunkSummaryLine::test_keys_are_sorted_for_stable_diffs — PEDANTIC — json.dumps(sort_keys) ordering; the parser reads keys by name
- TestChunkSummaryLine::test_carries_the_per_cell_run_id — DUP — the helper serialises whatever kwargs it is given; test_round_trips_as_json_with_stable_prefix pins that
- TestChunkSummaryLine::test_an_added_key_does_not_disturb_the_existing_ones — DUP — same round-trip
- TestPhaseParser::test_falls_back_to_legacy_prose_parsing — IMPROBABLE — PHASE_PARSER runs over SSM on LIVE workers; the prose fallback serves only workers running pre-2026-07-21 code, which no current image does. Delete the fallback branch in observe_cluster.py (~40 LOC of src) with it
- TestCoarsenedCompare::test_sampled_compare_dim_aware_without_time_axis — PARAM — fold into test_sampled_compare_selects_northing_rows as `with_time` param
- TestCoarsenedCompare::test_non_spatial_variable_compared_not_crashed — IMPROBABLE — a 0-d data var in a coarsened embedding store; a crash would be loud, not misleading
- TestCrossConfigEnvelope::test_a_zeroed_artifact_fails[False] — PARAM — the cross_config=True case is the stricter verdict path; one case pins cosine_min=0 → fail
Notes: kept tests are the ones whose failure would mislead an operator (phase-table columns, report reaching its last section, cross-config verdict thresholds, dim-aware row sampling). `_make_coarse_ds` and four TestCoarsenedCompare tests re-import numpy/xarray inline; hoist (both are already imported at top). Each TestPhaseParser case spawns a python subprocess (~0.1 s); one fewer.

## tests/unit/profiling/test_watch_scheduler.py — TRIM — tests 23→19, LOC 318→~270, cpu-s 0.0→0.0
- test_unknown_fleet_memory_is_json_safe — PARAM — same nan→None→strict-JSON path as test_parse_health_line_unknown_mem_is_json_safe; parametrise one test over the nan field
- test_measured_zero_spill_is_not_unknown — DUP — test_parse_health_line_full parses 6.5 as float and test_alert_worker_spill asserts 0.0 does not trip
- test_spill_trips_while_every_scheduler_signal_is_nominal — DUP — test_alert_worker_spill (the `_sample` defaults are already nominal)
- test_profile_run_empty — PEDANTIC — empty-input early return
Notes: everything else guards an operator-misleading failure (format drift in parse_health_line, alert thresholds, null-vs-zero peaks, the Insights runner's truncation and cancel contracts); keep.

## tests/unit/profiling/test_ingest_report.py — TRIM — tests 12→11, LOC 183→~170, cpu-s 0.0→0.0
- TestTruncationIsNeverSilent::test_missing_truncated_key_is_treated_as_complete — DUP — byte-identical call and assertion to test_untruncated_dossier_has_no_partial_warning (the fixtures carry no `truncated` key)

## tests/unit/profiling/test_ingest_log_queries.py — TRIM — tests 34→22, LOC 77→~55, cpu-s 0.0→0.0
- test_no_sort_on_a_raw_bin_expression[*] (11 ids) — DUP — test_sorted_fields_are_fields_the_query_defines captures `bin` from `sort bin(` and fails it as undefined
- test_every_query_has_a_description — PEDANTIC — non-empty docstring-like strings in a constant dict

## tests/unit/profiling/test_observe_cluster_parser.py — DELETE — tests 1→0, LOC 59→0, cpu-s 0.0→0.0
- test_phase_parser_valid_px_from_cdone — IMPROBABLE — exercises only the legacy prose-log path of PHASE_PARSER (its own docstring says so), which serves workers on pre-2026-07-21 code; current runs use the CHUNK_SUMMARY path tested in test_profiling_tools.py::TestPhaseParser. Goes with the fallback's removal from observe_cluster.py.

## Cross-file
- Src that only tests reach, recommended for deletion alongside its tests: inference/shard_record.py (whole module), sampling.resampler_for, sampling.bucket_for_count, quantization.quantize_embeddings + dequantize_embeddings, observe_cluster.PHASE_PARSER's legacy prose branch (~40 LOC). Owner call: student_v2.QKNormEncoderLayer (unreachable: only family sets enable_qk_norm=False).
- test_gpu_starvation.py is a campaign cost/starvation MODEL, not a test of src; relocate to scripts/ or context_docs/campaign/ (1,482 LOC out of the suite) and repoint 5 doc references.
- Layer duplication: strip reassembly is tested at the loader (test_data_loading TestStripReassembly) AND end-to-end at the actor (test_strip_loop single-vs-multi bit-identity); keep the actor test as the guarantee, the loader one only as a localiser. compute_doy is tested in both test_data_loading and storage/test_zarr_store. Quantization CPU/torch parity in test_quantization and test_student_v2.
- Source-text tests (`inspect.getsource` + substring/AST) appear in test_inference_actor (3) and test_sampling_v2_parity (2); the substring ones are brittle in both directions and are cut; the AST one (un-narrowed config) is kept.
- The two golden files (skip without TESSERA_V11_CKPT / TESSERA_V2_CKPT) earn their LOC: zero default cost, the only real-weights evidence of port fidelity. Optional merge into one parametrised module (~70 LOC).
- Biggest time lever in this slice is test_strip_loop (real InferenceActor.process_chunk per test, 2-3 runs per bit-identity test): cache the serial reference write in a module-scoped fixture, copy inference_config per test. Second is test_data_loading's real-icechunk concurrency test (shrink to 64² × 3 repeats). Third, the exhaustive 8,451-pair loops in test_sampling_v2_parity (keep one of three).
- Repeated scaffolding worth one helper each: test_scheduling's `config = MagicMock(); checkpoint_path=...` + ordered fake_wait/fake_get (~120 LOC); test_strip_loop's five synthetic-S2-store builders and five `_run` wrappers (~150 LOC); `ray.nodes` patches in test_scheduling are dead (src never calls it).
- Slice totals: tests 587→~400, LOC ~11,080→~7,500 (1,482 of it relocated, not lost), cpu-s ~22→~14.
