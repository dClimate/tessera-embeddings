# Report: tests/unit/assembly + tests/unit/orchestration/runners

## ROOT CAUSE OF SLOWNESS (applies to every slow file in this slice)
`storage/shard_writer.py::run_forked` (line ~513) builds a FRESH `ProcessPoolExecutor(mp_context=get_context("spawn"))` on EVERY
`ZarrWriter.assemble` / `assemble_global` / `write_year_shards` call, even for one payload (n_workers=1). Each spawned child
cold-starts Python and re-imports `tessera_embeddings` (~0.8 s of imports, 740 ms of it `ingest.s1_roi` pulled in by the package
`__init__`). Measured on the test_plain_update `area` fixture (96x128 px store, one assemble): **1.23 s with spawn vs 0.12 s
with an in-process thread pool** (10x); refusal tests that stop before the fork run in 0.1 s while one-fork tests take 2-6 s
under xdist. The icechunk/zarr work itself is cheap at test sizes. So "real store per test" is NOT the cost; "spawned pool
per assemble" is.
**Fix (one place, ~20 LOC, tests/unit/conftest.py autouse fixture):** monkeypatch `shard_writer.ProcessPoolExecutor` with a
`ThreadPoolExecutor` subclass that accepts `mp_context/initializer/initargs`, and whose `submit(fn, payload)` passes
`pickle.loads(pickle.dumps(payload))` — this keeps per-worker fork independence (each worker gets its own unpickled
ForkSession, exactly as spawn does) AND keeps the picklability check. Set `shard_writer._PROGRESS_SLOTS = initargs[0]` in the
shim (not `_init_fork_worker`, which calls `configure_logging()` and would clobber caplog). `_terminate_pool` already tolerates a
missing `_processes`. Opt-out marker (`@pytest.mark.real_spawn`) for the few tests whose subject IS the process pool:
test_shard_writer TestWorkerProgressReporting / TestRunForkedCatchUp / TestForkTelemetry / stall-watchdog tests, and one
assemble with n_workers>1 (test_assemble_multiband_parallel_matches_serial). Expected: assembly/ 200 -> ~40 cpu-s,
runners zone_fill/zone_update/plain_update 124 -> ~25 cpu-s, i.e. ~250 of the suite's 618 cpu-s, before any test is cut.
Module-scoped read-only stores add little on top once spawn is gone (the store build is ~0.1 s); use them only where noted.

## tests/unit/assembly/test_assembly.py — TRIM — tests 134→~104, LOC 3142→~2450, cpu-s 128→~65 (cuts) →~20 (with spawn shim)
- TestStagedStorageOptions::test_non_s3_path_gets_no_options[file://, relative] — PARAM — keep the /tmp id; same branch.
- TestWriteChunk::test_write_chunk_with_obs_counts — DUP — TestAssembly::test_assemble_with_obs_counts stages and reads back the same three vars.
- TestWriteChunk::test_write_chunk_without_obs_counts_backward_compat — PEDANTIC — field absence; TestAssembleGuards::test_append_creates_missing_variables_from_layout asserts it end to end.
- TestAssembly::test_a_store_without_the_record_takes_the_next_writes_stack — DUP — test_conventions::test_the_store_root_decides_the_stack_a_write_may_carry (unstamped root row) + the first-write stamp test here.
- TestAssembly::test_a_second_encoder_cannot_append_to_a_store_the_first_published — DUP — same assert_stack_matches call wired by test_the_store_keeps_its_first_writes_stack...; v2-on-v1.1 row in test_conventions.
- TestAssembly::test_a_store_with_no_recorded_encoder_is_read_as_v11_not_waved_through — DUP — test_conventions::test_one_comparison_serves_every_reader (None→v1.1 both ways). Slowest test in suite (5.9 s).
- TestAssembly::test_the_same_encoder_appends_normally — DUP — test_assemble_appends_to_existing_store (same family, time axis 2).
- TestAssembly::test_assemble_xarray_readable — DUP — fold its dims/coords asserts into test_assemble_multiple_chunks.
- TestAssembly::test_assemble_with_std — DEAD — config/inference.py:987 forces compute_std=False; no src/scripts caller passes True or embeddings_std.
- TestAssembly::test_assemble_append_with_std — DEAD — same.
- TestAssembly::test_same_date_overwrite_resets_untouched_vars_to_fill — DEAD — only embedding_std can be "dropped" between same-date runs; move its `_assemble-wip` branch assert into test_assemble_same_date_overwrites_in_place.
- TestAssembly::test_emits_one_assembly_summary_record — PEDANTIC — byte/write counts of a log record; global variant (kept) pins the record semantics.
- TestAssembly::test_assemble_sparse_roi_scales_nan_filled_for_non_live — DUP — add the scales-NaN assert to test_assemble_sparse_roi_non_live_chunks_are_zero_filled.
- TestCleanupStaging::test_only_the_target_run_is_deleted — DUP — test_delegates_to_delete_prefix_with_run_target asserts the exact prefix; merge all_versions assert into it too (3→1).
- TestSingleRoiChainIsAligned::test_outside_the_roi_stays_at_fill — DUP — test_assemble_sparse_roi_non_live_chunks_are_zero_filled; this one re-assembles 3000 px per test (2.5 s).
- TestVariablesAddedToAnExistingStore::test_a_fresh_store_still_gets_the_current_preset — DUP — TestSingleRoiChainIsAligned::test_output_geometry_matches_the_campaign.
- TestAssemblyValidation::test_assemble_root_attrs — PEDANTIC — attr existence/echoes of inputs (total_y, run_id).
- TestScanExistingStagedChunks::test_empty_staging_dir — PARAM — merge with test_no_staging_dir (same empty-set result).
- TestScanExistingStagedChunks::test_invalid_shape_raises — DUP — test_reports_all_invalid raises the same error on two such tiles.
- TestVerifyStagedCompleteness::test_no_staged_chunks_raises — DUP — test_missing_chunks_raises.
- TestVerifyStagedCompleteness::test_skip_marker_clears_a_stale_done_marker — DUP — merge into test_skip_marker_clears_a_stale_staged_zarr (one write_skip_marker, assert both gone).
- TestAssembleGlobal::test_a_run_off_the_stores_stack_is_refused_before_anything_lands[stack1, stack2] — PARAM — reasons are test_conventions rows; keep stack0 for the wiring + "nothing lands".
- TestAssembleGlobal::test_the_forking_write_asks_icechunk_to_cache_the_credential[True] — PARAM — keep [False], the case the docstring says matters.
- TestAssembleGlobal::test_a_skipped_tile_is_indistinguishable_from_one_never_written — DUP — test_a_retry_that_skips_a_tile_clears_the_previous_attempt asserts zero emb + NaN scales on the cleared tile.
- TestAssembleGlobal::test_every_requested_worker_forks — HISTORY — the removed S3 budget clamp; test_emits_one_assembly_summary_record (global) already asserts workers_used == partition.
- TestAssembleGlobal::test_month_coverage_survives_the_real_staging_path[s1_asc, s1_desc] — PARAM — same bool→int8 route per sensor; keep s2; values of all three are covered by test_every_carried_var_reaches_the_destination.
Notes: slowness = the spawn pool per assemble (see top); every TestAssembly/TestAssembleGlobal/TestAssembleGuards test that reaches the fork costs 1.5-6 s and ~0.1 s without it. TestSingleRoiChainIsAligned already shares a class-scoped store (good). test_coordinates_in_output writes a 1500x1500x128 float32 tile (0.6 s) to assert coordinate ranges — shrink to 20x20. TestPlainZarrIsReadWithTheRunsCredentials tests storage/zarr_store.plain_zarr_storage_options → MERGE INTO tests/unit/storage/test_zarr_store.py; test_radar_coverage_counts_fully_free_tiles_from_a_generator → MERGE INTO test_radar_coverage_record.py. src follow-up: ZarrWriter.write_chunk(embeddings_std=...) and assemble(compute_std=...) are dead code paths.

## tests/unit/assembly/test_shard_writer.py — TRIM — tests 99→~79, LOC 1812→~1560, cpu-s 71.5→~47 (cuts) →~10 (with spawn shim)
- test_a_source_that_owns_some_pixels_merges_into_the_shard[2] — PARAM — one live shard, so n_workers=2 adds no merge behaviour; keep [1] (4.2 s).
- test_years_complete_updated_in_commit — DUP — TestTheDrillDeathBetweenTheTwoCommits::test_an_unarmed_fill_cannot_reach_the_exit_at_all asserts years_complete==[2025].
- test_other_years_stay_empty — DUP — one assert; fold into test_writes_one_shard_land_masked (saves a seed + spawn).
- TestForkTelemetry::test_no_telemetry_param_changes_nothing — PEDANTIC — asserts return type is a str.
- TestForkTelemetry::test_run_forked_returns_worker_stats_in_payload_order — DUP — single payload; TestWorkerProgressReporting::test_run_forked_ships_index_and_interval_to_workers asserts the same workers list.
- TestWorkerProgressReporting::test_write_year_shards_threads_log_and_unit_through — MOCK — spies kwargs handed to run_forked; unit noun is wording.
- TestForkProgressReporting::test_the_progress_line_names_what_is_being_assembled[None] — PEDANTIC — default prefix wording; keep [15N].
- TestForkProgressReporting::test_progress_line_names_the_callers_unit — PEDANTIC — log noun wording.
- TestCommitTracing::test_a_rejected_directive_is_not_recorded_as_the_base — DEAD — set_base_logs_filter has no src/scripts/tests-infra caller (grep).
- TestCommitTracing::test_the_base_filter_is_applied_and_is_what_a_scope_returns_to — DEAD — same.
- TestCommitTracing::test_an_icechunk_without_the_hook_is_tolerated — IMPROBABLE — icechunk is pinned in the lock and has set_logs_filter.
- TestTheStalledCommitAlarm::test_the_filter_keeps_a_baseline_for_every_other_target — PEDANTIC — asserts a constant string.
- TestCatchUpToBranch::test_refuses_to_skip_a_commit_that_touched_our_own_zone — DUP — add its "base unchanged" assert to test_a_blocked_catch_up_leaves_the_collision_for_the_commit_to_raise.
- TestRunForkedCatchUp::test_the_tally_reaches_the_telemetry — DUP — test_the_tally_separates_the_outcomes asserts current>=1 too.
- TestRunForkedCatchUp::test_the_tally_reaches_the_summary_record_not_just_the_dict — PEDANTIC — substring match on inspect.getsource.
- TestRunForkedCatchUp::test_the_timer_fires_repeatedly_during_the_fork_phase — DUP — test_every_catch_up_happens_before_the_merge asserts >=2 ticks on the same single-payload path.
- TestRunForkedCatchUp::test_a_catch_up_that_will_not_stop_aborts_rather_than_merging — DUP — body identical to test_a_wedged_catch_up_fails_the_fill_rather_than_merging.
- TestRunForkedCatchUp::test_a_failing_catch_up_is_not_swallowed — DUP — test_a_timer_failure_reaches_the_caller (raise-once) is strictly stronger.
- TestForkStallWatchdog::test_a_stall_reads_as_a_stall_not_as_a_failed_catch_up — PEDANTIC — class hierarchy.
- TestForkStallWatchdog::test_the_production_timeout_is_far_outside_a_healthy_gap — PEDANTIC — constants.
Notes: slow because every write_year_shards/run_forked call spawns a pool (1.5-3 s each); `_seed` (real global repo, function scope) is ~0.1 s and not worth sharing. With the shim, keep real spawn only for test_run_forked_ships_index_and_interval_to_workers (picklability/pool smoke). TestDiffTouches could be one parametrized test (−40 LOC). Misfiled: TestCommitTracing + TestTheStalledCommitAlarm test storage/icechunk_logging, TestDiffTouches/TestCatchUpToBranch/TestCatchUpRaceAndBestEffort/TestTheCatchUpIsActuallyWiredIn (minus the first test) test storage/session_catch_up → move to tests/unit/storage/test_icechunk_logging.py and test_session_catch_up.py; test_every_assembly_commit_goes_through_the_traced_helper is an AST rule → tests/architecture. src follow-up: delete set_base_logs_filter (dead).

## tests/unit/orchestration/runners/test_zone_fill.py — TRIM — tests 58→49, LOC 1348→~1200, cpu-s 62→~53 (cuts) →~15 (with spawn shim)
- test_all_ocean_cell_marked_complete_without_inference — DUP — test_all_ocean_cell_skips_missing_mosaic is the stronger case; move the runs/tag asserts there.
- test_an_unusable_assembly_worker_override_is_refused_before_inference[-16] — PARAM — same branch as [-1]; keep [0] and [-1].
- test_all_tiles_skipped_marks_complete_empty — DUP — test_an_all_skipped_retry_clears_the_previous_attempt_s_shards ends in the same all-skipped state; add its tag + no-optical_skips asserts there.
- test_staging_cleanup_still_runs_inline_without_a_deferral — DUP — test_fill_zone_year_end_to_end asserts staging is gone after an undeferred fill.
- test_the_flow_actually_defers_both_deletes — PEDANTIC — substring match on inspect.getsource of the flow.
- test_zone_has_live_tiles_true — DUP — test_zone_live_tile_count asserts zone_has_live_tiles both ways.
- test_zone_has_live_tiles_false — DUP — same.
- test_non_calendar_year_windows_are_rejected[partial] — DUP — test_calendar_gate_is_callable_before_provisioning tests the partial window on the same function; keep [wrong-year] (wiring) and [rolling] (own message).
- test_infer_phase_threads_retirement_gate — MOCK — asserts a kwarg reached the stubbed run_inference.
- test_zone_year_complete_reflects_years_complete — TRIM (not delete): drop its full fill (2.3 s); test_phase_split_matches_composed_fill already asserts zone_year_complete is True after landing.
Notes: slow tests are exactly those that reach `assemble_global` (spawned pool, 2-4.6 s); tests that stop before it cost 0.1-0.35 s INCLUDING `_seed_global` + `_make_mosaic` (3 icechunk stores) + `_make_mask`, so store builds are not the problem. Spawn shim is the fix. Secondary: `_make_mosaic(tmp_path)` with default args is read-only and built by ~20 tests → a module-scoped `zone_mosaic` fixture (tmp_path_factory) saves ~0.1 s each; `_seed_global` must stay per-test (every test writes). test_wrong_grid_sar_store_raises / test_mosaic_grid_mismatch_raises / test_non_affine_mosaic_axis_raises cover three distinct guards — keep.

## tests/unit/orchestration/runners/test_zone_update.py — TRIM — tests 11→10, LOC 285→~270, cpu-s 32.6→~21 (cuts + module fixture) →~6 (with spawn shim)
- test_an_update_on_another_stack_is_refused_and_writes_nothing — DUP — the update writes via assemble_global → claim_root_stack; TestAssembleGlobal::test_a_run_off_the_stores_stack_is_refused_before_anything_lands[stack0] pins refusal + nothing lands.
Notes: slow because the function-scoped `filled` fixture runs a full `_fill` (seed + 3 mosaic stores + mask + inference stub + `assemble_global` with a spawned pool) for every test: 2.3-4.8 s each. Four tests only read it and are refused before inference (test_a_clearing_update_refuses_a_target_the_mask_has_no_land_in, test_a_holes_update_refuses_the_radar_rule, test_an_update_refuses_a_year_the_campaign_never_filled, test_an_out_of_grid_shard_is_refused_before_inference) → give them a module-scoped `filled_readonly` (tmp_path_factory + pytest.MonkeyPatch.context()) built once; saves ~7 s today, ~1 s after the shim. The rest write and need their own copy. Imports six private helpers from test_zone_fill — move `_seed_global/_make_mask/_make_mosaic/_staging_inference_stub/_fill/_config` into a runners/conftest.py or zone_fixtures.py helper module (no LOC change, removes a test→test import).

## tests/unit/orchestration/runners/test_plain_update.py — TRIM — tests 10→9, LOC 135→~130, cpu-s 29.3→~8 (module fixture) →~2 (with spawn shim)
- test_an_update_on_another_stack_is_refused — DUP — _write_update calls writer.assemble_global (plain.py:357) → claim_root_stack; same surviving test as above.
Notes: the function-scoped `area` fixture assembles a real single-area store (ZarrWriter.assemble → spawned pool, measured 1.23 s alone, 2.3-3.5 s under xdist) for every test. All 7 `test_an_update_that_cannot_be_planned_is_refused` params only call `_plan_update` (read-only, 0.01 s measured) → move them onto a module-scoped `area_readonly` fixture (tmp_path_factory); that alone removes ~19 cpu-s. The two writing tests keep the function-scoped `area`. The "mode must be" param duplicates storage/test_update.py:40 only in message, not in code (plain.py:300 is its own check) — keep.

## tests/unit/orchestration/runners/test_sequential_fill.py — TRIM — tests 77→60, LOC 2249→~1850, cpu-s 4.4→~3.5
- test_ingest_lifecycle_and_cleanup_after_assembly — DUP — test_ingest_starts_for_every_cell_and_is_not_paced_by_the_fill asserts every start and cleanup.
- test_assembly_failure_recorded_run_continues — DUP — test_a_failed_assembly_never_submits_a_delete raises the same "1/2 cell" and asserts more.
- test_a_cell_with_a_different_orbit_streams_under_its_own_orbit — DUP — test_the_streamed_work_carries_each_cell_s_own_orbit proves the cell streams with its orbit; the "deferred key gone" assert is HISTORY.
- test_every_cell_mismatching_the_session_orbit_still_completes — HISTORY — orbit deferral and its mosaic budget are deleted; orbit now only rides on ZoneContext.
- test_a_mismatched_run_does_not_deadlock_the_feeder — HISTORY — same removed deferral/slot mechanism.
- test_failed_cell_releases_budget_slot_and_keeps_mosaic — HISTORY — budget slots are gone (ingest ungated); retention is test_failed_zone_keeps_mosaic_and_others_land.
- test_session_crash_unwinds_feeder_without_deadlock — HISTORY — the slot semaphore it targets is gone; crash unwind is test_stop_unblocks_a_running_ingest_wait.
- test_feeder_crash_is_surfaced_not_silent_success — DUP — test_a_crashed_feeder_releases_the_stream_instead_of_holding_it, same inputs plus a deadline.
- test_the_feeder_takes_a_landed_cell_over_its_head_cell — DUP — test_the_readiest_cell_is_taken_wherever_it_sits is the general case.
- test_the_first_zone_is_handed_over_before_anything_is_withheld — DUP — identical run to test_withholding_starves_the_stream_and_then_lets_it_finish; fold its one ordering assert there.
- test_a_paused_poll_never_reports_exhaustion — DUP — test_a_pause_holds_the_stream_without_ending_it asserts the paused polls are empty, not none.
- test_no_pause_callable_means_no_check_at_all — DUP — every other test runs with paused=None; its assert (`"empty" not in seen[:1]`) is weak.
- test_the_cap_is_not_derived_from_look_ahead — PEDANTIC — asserts a default (100); the explicit-cap half is test_systematic_failure_stops_feeder_at_retained_cap.
- test_exceeding_the_cap_alerts_but_does_not_tear_the_run_down — DUP — asserts only waited==cap, as test_systematic_failure_stops_feeder_at_retained_cap; no alert is asserted.
- test_the_cap_alert_fires_for_a_failure_that_arrives_after_the_feeder_finishes — DUP — asserts only that a failing run raises; the alert it names is never checked.
- TestDrainTrailingAssemblies::test_a_wedge_is_a_typed_error_the_flow_can_tell_apart — PEDANTIC — class hierarchy and str.
Notes: fast already (pure fakes, threads). Not a time target; the win is LOC. Many docstrings narrate PR/review history ("THE PR #167 REVIEW FINDING", dates) — trim when touching. `_run_with_deadline` duplicates tests/unit/deadline.run_under_deadline — use the shared one (−20 LOC).

## tests/unit/assembly/test_skip_registry.py — TRIM — tests 81→61, LOC 1378→~1050, cpu-s 1.5→1.2
- TestTheMarker::test_a_zero_byte_marker_from_before_the_registry_is_not_an_error — DUP — test_an_unserialisable_record_costs_the_record_and_not_the_marker asserts an empty marker reads ({}, 0).
- TestTheSummary::test_units_are_in_the_field_names — PEDANTIC — key names.
- TestTheSummary::test_a_consistent_record_is_not_flagged — DUP — test_assembly TestAssembleGlobal::test_a_resolved_live_set_with_no_skips_records_a_zero asserts the whole dict.
- TestTheSummary::test_every_reason_the_dataset_computes_has_a_place_in_the_record — PEDANTIC — substring search of dataset.py source text.
- test_reading_many_markers_returns_every_one[1], [5] — PARAM — keep [50].
- TestTheGridAnAllSkippedRunWasStagedOn::test_markers_predating_the_field_still_report_unknown — DUP — test_assembly TestVerifyStagedCompleteness::test_detect_chunk_size_separates_all_skipped_from_no_such_run (record-less markers → AllChunksSkippedError).
- TestTheGridAnAllSkippedRunWasStagedOn::test_a_run_that_never_existed_still_raises — DUP — same test's typo-run half.
- TestTheDatasetIsActuallyReadable::test_an_unmeasured_tile_claims_no_refusal_measurement — DUP — TestAPartlyRefusedTileIsNotFullyCovered::test_zero_refusals_is_written_as_zero_not_null asserts the unmeasured row is None.
- TestTheDepthRuleTheRowsWereJudgedAgainst::test_a_row_states_the_line_its_own_refusal_was_measured_against — DUP — test_every_row_carries_it_including_embedded_ones.
- TestOneCellsRefusalsNeverReachAnother::test_a_second_assembly_does_not_inherit_the_first_cells_records — MOCK — the test performs the reset itself (`writer._last_skip_records = {}`), so assemble_global's reset is never exercised.
- TestOneCellsRefusalsNeverReachAnother::test_without_the_reset_the_stale_record_does_attach — MOCK — pins the hazard, not production code.
- TestPartsWrittenByDifferentBuilds::test_an_inferred_read_drops_a_column_the_older_part_lacks — PEDANTIC — asserts pyarrow's schema inference, not our code.
- TestPartsWrittenByDifferentBuilds::test_the_hazard_is_order_dependent_which_is_why_it_hides — PEDANTIC — same; keep test_stating_the_schema_recovers_it_and_nulls_the_older_part.
- TestRankingAnInfillByTheRightDepth::test_ranking_by_the_footprint_median_inverts_the_answer — DUP — ranks the test's own inputs; registry_rows only copies the two medians, pinned by test_the_two_medians_describe_different_populations.
- TestWhichBuildRefusedIt::test_no_build_information_at_all_is_null — PARAM — merge with test_a_wheel_install_with_no_commit_is_null_not_blank.
- TestWhereTheTileIs::test_an_antimeridian_row_keeps_the_west_greater_than_east_convention — DUP — registry_rows copies the tuple (test_every_row_gets_the_box_for_its_own_tile); its key assert is conditional (`if west > east`).
- TestTheEmbeddedHalfOfTheRecord::test_a_write_with_no_record_leaves_no_attribute — PEDANTIC — attribute absence.
- TestToppingUpTheRecords::test_nothing_missing_reads_nothing — PEDANTIC — guards a skipped round trip, not a result.
Notes: fast; LOC is the win. About half the file tests storage/registry.registry_rows and pyarrow parts (TestTheDatasetIsActuallyReadable, TestAPartlyRefusedTileIsNotFullyCovered, TestTheDepthRule..., TestPartsWrittenByDifferentBuilds, TestRankingAnInfill..., TestWhichBuildRefusedIt, TestWhereTheTileIs) → move to tests/unit/storage/test_registry.py; TestWhereTheTileIs::test_the_box_matches_the_shard_the_label_names tests zone_grid.tile_range_bbox_wgs84 → storage/test_zone_grid.py. Two `_write` helpers (TestTheDatasetIsActuallyReadable, TestTheDepthRule...) are near-identical — one helper.

## tests/unit/assembly/test_input_coverage_record.py — TRIM — tests 10→8, LOC 194→~150, cpu-s 0.1→0.1
- test_record_carries_input_coverage_when_supplied — DUP — test_a_relaxed_half_year_is_measured_and_says_it_was_relaxed asserts the field lands via run_provenance.
- test_absent_coverage_is_recorded_as_nothing — DUP — run_provenance treats every optional field with one `if x: record[k] = dict(x)` (shard_writer.py run_provenance); one parametrized absent-field test covers all four (see Cross-file).
Notes: test_a_legitimately_empty_month... and test_the_ingests_own_account... rebuild the MemoryStore inline; give `_serve` an `attrs=` kwarg (−15 LOC). Drop the `"assessed_unreadable_dates" not in store` HISTORY assert.

## tests/unit/assembly/test_radar_coverage_record.py — TRIM — tests 18→12, LOC 164→~120, cpu-s 0→0
- TestSummarisingAYear::test_a_fully_covered_year_reads_as_zero — DUP — test_the_percentage_is_of_embedded_area_not_of_the_zone covers the 0-free tile in its 50%.
- TestSummarisingAYear::test_the_thin_threshold_is_recorded_with_the_figure — PEDANTIC — echoes a constant.
- TestSummarisingAYear::test_an_empty_run_records_no_summary — DUP — test_a_year_that_embedded_nothing_records_no_summary (same zero-denominator branch).
- TestLocatingTheGapCoarsely::test_a_tile_with_no_radar_free_pixels_is_not_counted — DUP — its data never builds the "0 of 0" case its docstring names; same as test_a_partly_radar_free_tile_is_not_counted_as_wholly_free.
- TestWhereItIsRecorded::test_the_summary_lands_on_the_year_s_entry — DUP — test_each_year_keeps_its_own_summary.
- TestWhereItIsRecorded::test_recording_a_summary_does_not_disturb_the_rest_of_the_record — PEDANTIC — field existence.
Notes: receive test_assembly's test_radar_coverage_counts_fully_free_tiles_from_a_generator (same function).

## tests/unit/assembly/test_optical_skip_record.py — TRIM — tests 12→7, LOC 146→~95, cpu-s 0→0
- TestSummarisingAYear::test_the_live_total_the_count_is_against_is_recorded_with_it — DUP — fold `tiles_live` into test_the_count_and_the_labels_are_both_recorded.
- TestSummarisingAYear::test_no_skips_summarises_as_a_zero_not_as_nothing — DUP — test_assembly TestAssembleGlobal::test_a_resolved_live_set_with_no_skips_records_a_zero asserts the full dict.
- TestWhereItIsRecorded::test_the_summary_lands_on_the_year_s_entry — DUP — test_each_year_keeps_its_own_summary.
- TestWhereItIsRecorded::test_it_rides_alongside_the_radar_summary_not_instead_of_it — PEDANTIC — independent `if` branches; identity-field existence.
- TestIdempotency::test_re_recording_the_same_year_leaves_one_entry — DUP — test_re_recording_does_not_accumulate_labels (same replace path, stronger).
Notes: the kept unit tests here make two slow integration tests redundant — see Cross-file (TestAssembleGlobal::test_re_assembling_the_same_year_leaves_one_record 5.8 s, ::test_an_all_skipped_year_records_the_empty_flag_instead 2.5 s).

## tests/unit/assembly/test_code_identity_record.py — KEEP (minor) — tests 7→6, LOC 127→~115, cpu-s 0→0
- test_no_identity_means_no_field — DUP — fold into the shared parametrized run_provenance absent-field test (Cross-file).
Notes: test_the_record_carries_the_identity_without_the_caller_supplying_one patches environment.code_identity AND shard_writer.code_identity; only the second matters (−2 LOC).

## tests/unit/orchestration/runners/test_orchestration_concurrency.py — DELETE — tests 4→0, LOC 104→0, cpu-s 0.4→0
- all 4 tests — DEAD — `sliding_window_submit` (orchestration/concurrency.py, 92 LOC) has no caller in src/ or scripts/; only a docstring mention in tessera_embeddings/__init__.py. Delete the module too.

## tests/unit/orchestration/runners/test_plain_runner.py — TRIM — tests 6→3, LOC 136→~75, cpu-s 0→0
- TestRunIngestOrbitDecomposition::test_ascending_calls_ingest_s1_once / test_descending_calls_ingest_s1_once — MOCK/PARAM — call-count proxies for data_loading._active_orbits; replace the whole class with one direct parametrized test of `_active_orbits` (none/ascending/descending/both), which nothing tests directly today.
- TestRunIngestOrbitDecomposition::test_s2_always_called_once — MOCK — mock call count.
- test_the_run_id_prefix_helper_is_public — PEDANTIC — `__all__` membership; the values are asserted in flows/test_run_id_model_version.py:130.
Notes: test_the_runner_preflights_the_output_store_before_inferring (ordering of a real gate) is the one test here worth its LOC. Merge the survivors into test_plain_runner_wiring.py (same module under test).

## tests/unit/orchestration/runners/test_plain_runner_wiring.py — TRIM — tests 24→21, LOC 253→~235, cpu-s 0.1→0.1
- TestTheCoverageThresholdReachesIngest::test_an_out_of_range_threshold_is_refused[negative], [way-over] — PARAM — keep the boundaries [zero] and [just-over].
- TestTheStagingIdentityRepeats::test_the_identity_is_not_the_campaign_fingerprint — MOCK — asserts a patched function was not called; test_ingest_code_identity_is_excluded_from_the_digest pins the real property by value.

## Cross-file
- THE time fix is one autouse fixture in tests/unit/conftest.py replacing `shard_writer.ProcessPoolExecutor` with a pickling ThreadPoolExecutor shim (details at top); opt-out marker for the ~3 tests whose subject is the real pool. Measured 10x on one assemble (1.23 s → 0.12 s). Estimated ~250 of the suite's 618 cpu-s, CI legs ~2 min shorter. Secondary, independent of it: package import pulls `ingest.s1_roi` (740 ms) from `tessera_embeddings/__init__`, paid by every spawned child in production too — a lazy import there would also cut it.
- Module-scoped read-only fixtures: test_plain_update `area_readonly` (7 refusal params) and test_zone_update `filled_readonly` (4 refusal tests); ~26 cpu-s today, ~2 s after the shim. Everything else that builds a store writes to it, and the store build itself is ~0.1 s — not worth sharing.
- Additional cross-layer cuts (counted in the table): test_assembly TestAssembleGlobal::test_re_assembling_the_same_year_leaves_one_record (5.8 s) — DUP of test_optical_skip_record TestIdempotency (run_provenance replace semantics); TestAssembleGlobal::test_an_all_skipped_year_records_the_empty_flag_instead — DUP of test_optical_skip_record::test_an_empty_year_records_the_flag_instead_of_the_labels + test_zone_fill all-skipped retry; test_zone_fill::test_a_store_with_no_depth_rule_publishes_null_not_zero (3.6 s) — DUP of test_skip_registry TestTheDepthRule...::test_no_rule_stays_null_rather_than_becoming_zero (None is the default at every hop).
- Stack/encoder refusal is tested at 5 layers: conventions (unit, keep), single assemble (keep 1), assemble_global (keep 1), zone_update + plain_update (cut — same claim_root_stack).
- run_provenance optional fields (radar_coverage, optical_skips, input_coverage, code) are one `if x: record[k] = dict(x)` each; "lands"/"absent adds no key"/"each year keeps its own" are written 3-4 times across four files → one parametrized tests/unit/storage/test_run_provenance.py (−~80 LOC), with the empty-suppresses-optical and previous-history tests beside it.
- Misfiled: icechunk_logging + session_catch_up tests live in test_shard_writer.py; registry_rows/pyarrow tests live in test_skip_registry.py; plain_zarr_storage_options tests in test_assembly.py. Move, no LOC change.
- test_zone_update imports six private helpers from test_zone_fill → runners/conftest.py or a zone_fixtures.py helper.
- Helpers: mosaic_stores.py, coverage_repo.py, zone_density.py are not used by any file in this slice (users are ingest/, inference/, flows/, storage/); mosaic_stores is in-memory and coverage_repo's repo build is ~0.05 s, so neither is a slowness source. No helper becomes unused from these cuts.
- Dead src surfaced: orchestration/concurrency.py (sliding_window_submit), icechunk_logging.set_base_logs_filter, ZarrWriter embeddings_std/compute_std path (config forces compute_std=False).

| file | verdict | tests | LOC | cpu-s |
|---|---|---|---|---|
| assembly/test_assembly.py | TRIM | 134→102 | 3142→~2400 | 128.5→~20 |
| assembly/test_shard_writer.py | TRIM | 99→79 | 1812→~1560 | 71.5→~10 |
| runners/test_zone_fill.py | TRIM | 58→48 | 1348→~1190 | 61.7→~14 |
| runners/test_zone_update.py | TRIM | 11→10 | 285→~270 | 32.6→~6 |
| runners/test_plain_update.py | TRIM | 10→9 | 135→~130 | 29.3→~2 |
| runners/test_sequential_fill.py | TRIM | 77→60 | 2249→~1850 | 4.4→~3.5 |
| assembly/test_skip_registry.py | TRIM | 81→61 | 1378→~1050 | 1.5→1.2 |
| assembly/test_input_coverage_record.py | TRIM | 10→8 | 194→~150 | 0.1→0.1 |
| assembly/test_radar_coverage_record.py | TRIM | 18→12 | 164→~120 | 0→0 |
| assembly/test_optical_skip_record.py | TRIM | 12→7 | 146→~95 | 0→0 |
| assembly/test_code_identity_record.py | KEEP (minor) | 7→6 | 127→~115 | 0→0 |
| runners/test_orchestration_concurrency.py | DELETE | 4→0 | 104→0 | 0.4→0 |
| runners/test_plain_runner.py | TRIM (merge into wiring) | 6→3 | 136→~75 | 0→0 |
| runners/test_plain_runner_wiring.py | TRIM | 24→21 | 253→~235 | 0.1→0.1 |
| **total** | | **551→426** | **11812→~9240** | **~330→~57** |
