# Report: tests/unit/storage, tests/unit/config, root unit tests, tests/architecture

## tests/unit/storage/test_campaign.py — TRIM — tests 32→25, LOC 442→~360, cpu-s 60→~14
- test_status_pending_lists_gaps — DEAD — `CampaignStatus.pending` has no src/scripts caller (grep `\.pending(` empty); work_list reimplements it.
- test_status_pending_counts_unseeded_zone — DEAD — same; unseeded-zone case survives in test_work_list_defaults_to_all_120_zones.
- test_years_fully_complete — DEAD — `years_fully_complete` has no caller outside campaign.py (8.4 s, the slowest test).
- test_tag_zone_year_idempotent — DUP — test_tag_zone_year_default_is_idempotent_after_head_moves is the stronger form.
- test_year_complete_tag_cannot_be_minted_from_a_zone_subset — PEDANTIC — asserts a TypeError on a nonexistent kwarg (mypy's job); refusal survives in test_tag_year_complete_requires_all_zones.
- test_a_no_land_mark_composes_with_a_real_fill_of_another_year — DUP — mark_zone_year_empty delegates to commit_year_attrs; union proven by test_commit_year_attrs_retries_a_conflict_and_produces_the_union.
- test_the_attr_commit_is_separate_from_the_shard_commit — MERGE into test_two_years_of_one_zone_both_land_with_neither_mark_lost (fill twice; assert +4 commits and both years).
Slow cause (measured): `_fill` → `write_year_shards` → `run_forked` ALWAYS opens a spawn-context ProcessPoolExecutor, even for one payload, so every `_fill` pays a fresh interpreter + package import (~1.5-2 s); ~20 fills across the file. `_seed` is <0.5 s.
Fix: make `_fill` land a year via `campaign.mark_zone_year_empty(repo, zone, _YEARS[i])` (attrs-only commit, same `years_complete` writer) for every status/tag/work-list/gc test; keep the real `write_year_shards` only in the one merged two-years test. Removes ~19 process spawns.
src doc bug: `write_year_shards` docstring (shard_writer.py:903-904) says "in-process when 1, else spawned processes"; `run_forked` (line ~511) deliberately routes one payload through the pool. Fix the docstring.
The four commit_year_attrs/RebaseFailed tests are shard_writer tests living here; move them to tests/unit/assembly/test_shard_writer.py if that file keeps a section for commit_year_attrs.

## tests/unit/storage/test_zarr_store.py — TRIM (large src deletion behind it) — tests 70→21, LOC 1121→~330, cpu-s 40→~6
- TestRollbackCommits (all 6) — DEAD — `rollback_commits` has no caller in TE src/scripts or yield-embeddings src (grep). Delete the function too.
- TestPadRegionToChunks (all 13) — DEAD — `_pad_region_to_chunks` is reached only via `_write_region` ← `write_region`, which nothing calls (TE src/scripts, YE src all empty).
- TestWriteRegion (all 8, incl. test_unaligned_region_write_under_distributed_client 5.9 s) — DEAD — same; `write_region` has zero production callers.
- TestWriteRegionSharded (8 inherited + 2 own, incl. distributed 6.4 s) — DEAD — same.
- TestParallelAccess (all 4) — DUP/PEDANTIC — threads over separate repos test icechunk, not us; the S3 pair runs through `set_s3_config`, a test-only hook (no src caller). Isolation survives in TestIcechunkTransactions-free form below.
- TestIcechunkTransactions::test_uncommitted_write_not_visible — PEDANTIC — drives `to_icechunk` + a raw session directly; asserts icechunk's own isolation, not our code.
- TestIcechunkTransactions::test_committed_write_visible — DUP — TestReflectanceStore::test_append_extends_time_dimension.
- TestReflectanceStore::test_create_store_stores_crs — DUP — test_append_preserves_crs_attr asserts the crs after create+append.
- TestReflectanceStore::test_create_store_uses_expected_chunk_sizes — DUP — identical to TestWriteDataset::test_write_dataset_clamps_chunks_to_extent.
- TestCleanupOnFailure::test_successful_write_preserves_store — DUP — every write test asserts the store exists and reads back.
- TestComputeDoy::test_returns_int32_array — PEDANTIC — dtype constant.
- TestComputeDoy::test_non_leap_year — PARAM — fold 2023-12-31 into test_computes_correct_doy_for_known_dates.
- TestDefaultRepoConfig::test_no_cache_override — PEDANTIC — asserts we set nothing (equals icechunk default).
Slow cause: two `distributed.Client(processes=True, n_workers=2)` tests (12 s) plus `_make_region_store` writing 1000x1000x(N bands) stores per test (TestResolveRegion, TestWriteRegion*). Deleting the dead region-write classes removes both. Shrink TestResolveRegion's store to 100x100 (`_make_region_store` height/width) and build it once per class (class-scoped fixture; all its tests are read-only except test_non_contiguous_range_raises).
src deletions the cuts enable (~550 src LOC): `storage/region_writes.py` (305 LOC), `write_region` + `_write_region` (zarr_store.py:1028-1075, 1325-1365), `rollback_commits`, `set_s3_config` + `_s3_config_override` branch (zarr_store.py:439-450, 1988-1994) and `S3Config` (no other user). Update storage/README.md:43 which names region_writes.py. `resolve_region` stays: yield-embeddings domain/roi_merge.py calls it.
Move TestComputeDoy to a time_axis test (compute_doy lives in storage/time_axis.py) if one exists; otherwise leave.

## tests/unit/storage/test_published_store.py — TRIM — tests 48→37, LOC 478→~400, cpu-s 12→~8
- TestSampleLivePixels (all 4) — DEAD — `sample_live_pixels` is called only by scripts/diagnostic/published_store_read_bench.py (a one-off benchmark); move the function into that script.
- TestAnonymousStorage::test_an_installed_override_refuses_an_anonymous_read — DEAD — `_s3_config_override` is set only by the test-only `set_s3_config` (no src caller); delete with it.
- TestAnonymousStorage::test_an_installed_override_still_serves_a_credentialed_read — DEAD — same.
- TestSavedManifestPreload::test_tags_and_the_branch_tip_survive_the_switch — PEDANTIC — its own docstring: no mutation of this package turns it red; it tests icechunk.
- TestCoordinateDepartures::test_a_misplaced_axis...[easting-shifted-one-pixel], [easting-at-twice-the-pixel-spacing] — PARAM — same "easting: starts" check as northing-reversed; keep northing-reversed, easting-deleted, month-zero-based.
- TestCoordinateDepartures::...[month-reversed] — PARAM — same value comparison as month-zero-based.
- TestLayoutDepartures::test_one_broken_property...[month-coordinate-deleted] — PARAM — same branch as coordinate-deleted (the audit reads the seeder's coordinate set, so one deleted coordinate proves it).
Slow cause: function-scoped `seeded` seeds a full GLOBAL-layout zone (2x4 shards of 2048 px) per test (~0.25 s each); `_fill_shard` writes whole 2048² shards (embeddings 128 bands) — ~1 s per test that fills.
Fix: module-scoped `seeded_template` seeded once, per-test `shutil.copytree` into tmp_path (local icechunk is a directory); read-only tests (conforms, calendar years, no live shards, seeded matches grid) take the template directly.
TestSavedManifestPreload tests `global_store.set_saved_manifest_preload`: MERGE INTO test_global_store.py.

## tests/unit/storage/test_fix_published_store_spatial_transform.py — DELETE (if the repair has been applied; else TRIM) — tests 22→0 (TRIM: 22→16), LOC 282→0 (TRIM: ~230), cpu-s 12→0 (TRIM: ~5)
- whole file — HISTORY — tests a one-shot maintenance script (ADR 024, 2026-09-14) whose fix is now structural: the seeder writes the corner origin, guarded by test_global_store/test_conventions. Owner to confirm the published store was repaired (a dry run against it prints "nothing to do"); then delete the tests, and the script per scripts/README.md policy.
Fallback TRIM if the repair is still pending:
- TestTheRepair::test_the_origin_moves_to_the_bbox_corner — DUP — test_the_repair_restores_what_the_seed_would_have_written compares whole attrs to the seeder's.
- TestTheRepair::test_the_registration_urls_are_repinned — DUP — same.
- TestTheSeedIsAlreadyCorrect::test_a_fresh_store_gains_no_commit — MERGE into test_a_fresh_store_is_a_clean_no_op (one more assert).
- TestRefusals::test_a_rotated_transform_is_refused, test_a_shape_disagreeing_with_the_arrays_is_refused, test_an_unknown_zone_is_refused — PARAM — collapse the attr-corruption refusals into one parametrized test over (attr, value).
Slow cause: function-scoped `seeded` seeds two GLOBAL-layout zones per test (~0.5 s). Fix: module-scoped seed + `shutil.copytree` per test.

## tests/unit/storage/test_global_store.py — TRIM — tests 28→21, LOC 389→~320, cpu-s 9→~6
- test_sibling_groups_seeded_without_clobber — DUP — test_seeding_a_zone_that_already_exists_is_a_no_op asserts both groups after an incremental seed.
- test_second_seed_adds_groups_without_clobbering_first — DUP — same survivor.
- test_embeddings_is_sharded — DUP — test_each_model_seeds_its_own_spatial_names asserts `layout_departures(g) == []`, which checks chunks and shards.
- test_global_store_config_has_time_split_and_preload — PEDANTIC — field existence.
- test_reseed_with_same_geometry_but_different_dtype_rejected — PARAM — same layout comparison as test_reseed_with_different_shard_geometry_rejected; parametrize the pair over (layout override, match).
- test_reseed_with_a_different_minimum_depth_rejected — PARAM — same write-once identity check as test_reseed_with_different_model_rejected and test_adding_a_rule_to_a_store_seeded_without_one_is_rejected.
- test_the_minimum_depth_rule_is_stamped_on_the_root — DUP — test_an_unstamped_store_keeps_its_identity_when_nothing_is_created asserts `optical_min_obs == 15` on the root.
Slow cause: every test seeds GLOBAL-layout zones into a fresh repo (~0.3 s). Fix: one module-scoped seeded store for the four read-only tests (data_vars_are_metadata_only, coords_and_attrs, root_carries_geoemb_convention, group_aware_open_store_reads_one_zone).
Receives TestSavedManifestPreload (4 tests after cuts) from test_published_store.py.

## tests/unit/storage/test_store_manifest.py — TRIM — tests 62→43, LOC 587→~435, cpu-s 1.7→1.2
- TestManifestSerialization::test_roi_manifest_to_dict — PEDANTIC — field-by-field dict echo; test_from_dict_round_trip covers it.
- TestManifestSerialization::test_empty_ingest_manifest_to_dict — DUP — test_to_dict_excludes_none_values.
- TestManifestSerialization::test_base_class_is_store_manifest — PEDANTIC — isinstance.
- TestManifestHash::test_hash_is_deterministic — PEDANTIC — h()==h().
- TestManifestHash::test_hash_is_16_chars — PEDANTIC — length constant.
- TestValidateAgainst::test_mismatched_crs_raises, test_mismatched_chunk_size_raises — PARAM — same loop as test_mismatched_resolution_raises; test_multiple_mismatches_reported covers a second field.
- TestValidateAgainst::test_error_message_includes_store_path — PEDANTIC — message text.
- TestValidateAgainstLegacyStores::test_none_manifest_logs_store_path — PEDANTIC — log wording; keep test_none_manifest_warns_but_passes (drop its caplog assert too).
- TestValidateAgainstEdgeCases::test_extra_keys_in_existing_ignored — DUP — test_unknown_key_in_existing_ignored.
- TestValidateAgainstEdgeCases::test_current_concrete_vs_existing_none_passes — DUP — TestAbsenceIsAValueForAPolicyField::test_a_field_outside_the_policy_set_keeps_the_forward_compatible_rule (identical body).
- TestIngestManifestChaining::test_ingest_manifest_validates_roi_hash — DUP — test_matching_config_passes (self-validation).
- TestIngestManifestChaining::test_ingest_manifest_detects_roi_hash_mismatch — DUP — test_store_has_field_current_dropped_raises + generic mismatch tests; same validate loop.
- TestEmbeddingManifestChaining::test_embedding_manifest_validates_model_checkpoint — DUP — self-validation again.
- TestEmbeddingManifestChaining::test_embedding_manifest_detects_num_obs_checkpoints_change — PARAM — same loop as detects_model_change.
- TestIngestManifestFromRoiStore::test_roi_hash_changes_when_roi_config_changes — DUP — test_chains_upstream_roi_hash + test_hash_changes_with_different_values.
- TestEmbeddingManifestFromUpstreamStores::test_combines_both_sar_orbits — PEDANTIC — asserts not-None and length 16; order/collision tests carry the behaviour.
- TestIngestCodeIsPartOfMosaicIdentity::test_the_identity_is_deterministic — PEDANTIC — f()==f().
- TestAdmissionThresholdIsPartOfIngestIdentity::test_sar_stores_carry_no_threshold — DUP — test_the_field_does_not_move_an_existing_hash asserts `min_valid_coverage is None` on the same dict.
Notes: TestS2OnlyPolicyIsPartOfEmbeddingIdentity and TestAbsenceIsAValueForAPolicyField overlap (policy-off == legacy); keep both, one asserts the hash, the other validate_against.

## tests/unit/storage/test_registry_master.py — KEEP (2 cuts) — tests 57→55, LOC 586→~575, cpu-s 1.8→1.8
- TestProvenance::test_measured_by_names_who_measured_not_which_pass_wrote_the_row — DUP — TestPrecedence::test_a_tile_nothing_has_measured_stays_in_the_master_as_a_null_row, same input, same assert.
- TestProvenance::test_the_master_schema_is_the_registrys_plus_provenance — PEDANTIC — column order; TestUpdateCell::test_an_update_replaces_its_tiles... already checks the written schema.
Production-critical merge logic (fill path, `merge_cell`), table-level and fast; well parametrized.

## tests/unit/storage/test_conventions.py — TRIM — tests 49→38, LOC 537→~490, cpu-s 0.1→0.1
- TestTileIdToEpsg::test_valid_tiles[37PBM, 01NBJ, 19MCR, 36MYE] — PARAM — keep 33UWP (north), 56HKH (south), 60NUG (edge zone).
- TestTileIdToEpsg::test_invalid_tiles_return_none["", "AB", "00NBC", None] — PARAM — keep kenya_highlands, 61NBC, 37ABM (each a distinct rejection branch).
- TestBuildConventionAttrs::test_model_url_overrides_derived_public_ref — DUP — test_model_url_tracks_the_encoder_family asserts explicit URL wins; test_model_is_public_ref... asserts derivation.
- TestBuildConventionAttrs::test_epsg_code_without_tile_id — DUP — test_epsg_code_overrides_tile_id exercises the same epsg_code branch.
- TestBuildConventionAttrs::test_gsd_uses_explicit_value_when_supplied — DUP — TestMultiGroupPlacement::test_root_attrs_are_geoemb_only passes gsd=10.0 and asserts it.

## tests/unit/storage/test_store_write_retry.py — TRIM — tests 22→14, LOC 479→~390, cpu-s 0.1→0.1
- test_transient_failure_recovers_on_a_later_attempt — DUP — test_transient_failure_is_retried_to_the_attempt_limit's second assert proves recovery with an attempt count.
- test_every_failure_surfaces_as_itself_not_as_a_retryerror — DUP — each `pytest.raises(<type>)` already appears in the conflict, duplicate-date and transient tests.
- test_concurrent_writer_errors_names_both_collision_shapes — PEDANTIC — tuple membership.
- test_no_ingest_module_builds_its_own_retry_policy[s1_roi, s2_roi] — HISTORY — source-grep guard; enforce with a ruff TID251 banned-api entry for `tenacity.Retrying` (the repo already uses TID251), or drop.
- test_a_failure_the_caller_waits_out_outlasts_the_attempt_limit — DUP — fold `calls > STORE_WRITE_ATTEMPTS` into test_the_wait_out_budget_bounds_the_backoff_it_authorises (same `_exhausted` call).
- test_a_failure_the_predicate_declines_gets_only_the_attempt_limit, test_a_cause_that_was_stripped_in_transit_gets_no_long_wait — PARAM — same `_exhausted(..)[0] == STORE_WRITE_ATTEMPTS` shape; add both as ids to test_a_refusal_that_is_not_the_sources_gets_no_long_wait.
Note: the autouse `_capture` fixture uses `clear_local_refusals`, a src function with no src caller; move it into the test module (or conftest) as a fixture helper.

## tests/unit/storage/test_open_or_create.py — TRIM — tests 18→13, LOC 362→~310, cpu-s 0.3→0.25
- test_an_existing_repo_opens — DUP — byte-for-byte the body of test_a_rootless_repo_opens_rather_than_creates.
- test_a_non_absence_failure_is_reraised_and_nothing_is_created[expired-credential, disk-full] — PARAM — one `not is_missing_repo` branch; keep throttle and real-corruption.
- test_the_original_exception_object_propagates — DUP — fold `caught.value is exc` into the parametrized reraise test.
- test_a_create_failure_is_not_caught_by_this_function — PEDANTIC — pins control-flow shape, not an observable behaviour.
Notes: test_a_dirty_prefix_still_raises_corrupted_with_an_actionable_message asserts four message phrases (PEDANTIC); keep the raise and the `__cause__` check only. test_the_ingest_commit_path_never_rebases has inline imports of `inspect` and `zarr_store` (already imported at module top).

## tests/unit/storage/test_object_store.py — KEEP (1 cut + dedupe fakes) — tests 16→15, LOC 336→~280, cpu-s 0→0
- test_delete_prefix_forwards_all_versions_when_asked — DUP — test_delete_prefix_defaults_to_the_safe_choice_not_the_cheap_one asserts `all_versions=True` reaches s5cmd.
Notes: three hand-written `_FS` fakes (test_delete_prefix_strict_raises_when_delete_fails, test_a_missing_binary_is_not_retried, test_strict_all_versions_refuses_the_fsspec_fallback) duplicate `_fallback_fs`; extend `_fallback_fs` with an `rm` hook and reuse it (~40 LOC). The middle block of test_delete_prefix_strict_raises_when_delete_fails (strict+all_versions refused) duplicates test_strict_all_versions_refuses_the_fsspec_fallback; drop it.

## tests/unit/storage/test_registry_rebuild.py — KEEP (4 cuts) — tests 31→27, LOC 321→~285, cpu-s 0.2→0.2
- TestCompareRow::test_eligible_px_alone_never_fails_the_gate — DUP — identical call and inputs to test_a_faithful_rebuild_matches.
- TestDriftAgainstTheFill::test_the_refusal_split_matches_the_datasets_rule — DUP — recomputes the same expressions locally; test_partitions_the_tile_into_zero_thin_and_deep pins the split, test_optical_statistics_match_the_actors_coverage_record pins the fill.
- TestMeasurementsFromObs::test_radar_presence_is_null_when_not_measured — DUP — TestReadingTheStore::test_rebuild_tile_reads_the_right_window_and_year asserts `px_with_any_radar is None`.
- TestRebuiltRow::test_a_run_id_says_it_is_a_rebuild — PEDANTIC — string prefix.
`rebuild_tile`/`as_coverage_record` are live via runners/zone_fill.py; `rebuilt_row`, `basis_violations`, `rebuild_run_id` are reached only by scripts/maintenance/rebuild_registry_measurements.py (a reusable repair/verify tool), so their single tests stay.

## tests/unit/storage/test_rebuild_registry_measurements_script.py — TRIM — tests 20→9, LOC 309→~170, cpu-s 1.6→0.8
The script's real job (re-deriving v1.1's unmeasured rows) is done; what stays live is its `verify`/`repair` CLI plumbing. The merge/audit/store-check logic it calls is already pinned in test_registry_master.py, so script tests that re-assert a library verdict go.
- TestRepair::test_an_optical_only_repair_cannot_undo_a_radar_measurement — DUP — test_registry_master TestPrecedence::test_a_rebuild_that_measured_radar_wins[a-later-optical-only-pass-cannot-undo-it].
- TestRepair::test_a_master_file_whose_last_part_was_withdrawn_is_emptied — DUP — test_repair_reaches_a_zone_with_no_part_left covers discovery from the master file; emptying is TestUpdateCell::test_a_cell_whose_last_part_is_withdrawn_is_emptied.
- TestRepair::test_a_cell_the_store_cannot_stand_behind_is_left_as_it_was[both] — DUP — TestConcurrentWriters::test_a_cell_the_store_cannot_stand_behind_is_not_written.
- test_a_refill_that_never_published_blocks_only_while_it_holds_a_tile[both] — DUP — TestTheStoreStandsBehindTheParts::test_a_run_with_no_part_that_last_wrote_a_tile_is_refused / ..._later_runs_wrote_over_does_not_block.
- TestVerify::test_a_damaged_master_fails[both] — DUP — TestAudit::test_damage_is_reported_against_its_cell (same two damages, library level).
- test_a_registry_without_parts_fails_rather_than_passing_vacuously[gate, verify] — PARAM — one shared guard; keep [repair].
- test_a_zone_the_registry_does_not_hold_fails[gate] — PARAM — keep [repair]; TestVerify::test_zones_narrow_what_is_checked already asserts 99X for verify.
Survivors: dry run writes nothing, null rows re-derived + verify passes, zone with no part, first fill never published (2), zones narrow, sound master verifies, two guards.

## tests/unit/storage/test_update.py — KEEP (1 cut) — tests 35→34, LOC 297→296, cpu-s 0.4→0.4
- test_a_plan_can_come_from_the_registry[any-None-cells1] — PARAM — fixture rows set refused_px == refused_thin_px, so "any" cannot differ from "thin"; either give the "any" row a non-thin refusal or drop it.

## tests/unit/storage/test_write_day_windows.py — TRIM + MERGE (absorbs test_write_days_windows_batched.py) — tests 10→8, LOC 295→~260, cpu-s 2.3→1.9
- test_manifest_mismatch_fails_before_any_write — DUP — test_an_override_append_records_both_code_identities_on_the_store asserts the same refusal; move its "nothing committed" assert there.
- test_parallel_windows_matches_sequential — DUP — test_write_days_windows_batched::test_batch_matches_singles_byte_for_byte[overlapped] compares sequential singles against the overlapped path.
Both paths are live: ingest flows default `overlap_window_writes=True`, the sequential path is the documented fallback. test_parallel_falls_back_when_private_api_missing has an inline `import sys`.

## tests/unit/storage/test_write_days_windows_batched.py — MERGE INTO tests/unit/storage/test_write_day_windows.py — tests 7→6, LOC 173→~100 after merge, cpu-s 2.9→2.7
- test_duplicate_date_in_batch_refused — PARAM — same "strictly increasing" guard as test_out_of_order_dates_refused_before_any_write; parametrize the pair.
Both files define identical `CHUNKS`, `GEOBOX`, `MANIFEST`, `_Roi`, `_day_ds`, `_snapshots`, and near-identical write helpers; one module saves ~70 LOC. test_failed_batch_commits_none_of_its_dates has an inline `import ... zarr_store as zs`.

## tests/unit/storage/test_empty_store.py — TRIM — tests 13→10, LOC 233→~185, cpu-s 1.1→0.9
- test_both_creation_paths_store_the_mosaic_codec — MERGE into test_grid_identity_matches_real_write (identical setup; add the compressor assert to its per-var loop).
- test_coords_match_geobox — DUP — test_grid_identity_matches_real_write compares coords to a write built from the same geobox.
- test_footprint_is_metadata_scale — DUP — test_no_data_chunk_objects asserts `nchunks_initialized == 0`, the direct form.
Not dead: `daily_times` (on the shared-context dead list) is imported by yield-embeddings domain/empty_store.py and used by its create_empty_stores flow; keep test_daily_times. test_creation_is_extent_independent asserts wall time < 5 s on a 40,000² store: a latent CI flake; it could assert `nchunks_initialized == 0` at that extent instead.

## tests/unit/storage/test_batched_region_writes.py — TRIM — tests 7→5, LOC 158→~120, cpu-s 0.5→0.4
- test_appended_date_reaches_the_dedupe_reader — DUP — test_failed_batch_commits_nothing and test_write_day_windows both assert `get_existing_dates` after appends through this writer.
- test_the_refusals_name_the_store_and_the_date — PEDANTIC — message wording.
Also drop the `issubclass`/`CONCURRENT_WRITER_ERRORS` asserts in test_duplicate_date_refused (PEDANTIC).

## tests/unit/storage/test_destination_preflight.py — KEEP — tests 11→11, LOC 114, cpu-s 0.1. (Drop the two message-wording asserts in test_the_bool_month_array_that_cost_two_fills_is_refused: PEDANTIC.)

## tests/unit/storage/test_zone_grid.py — TRIM — tests 146→20, LOC 87→~75, cpu-s 0.3→0.3
- test_extents_pinned_to_pyproj[×120] — PARAM — collapse to one test looping over `zg.ZONES` (same coverage, one id; collection and report noise drop by 119 items).
- test_shard_aligned[×6] — PARAM — fold the `% SHARD_PX` asserts into that same loop over all 120 zones (stronger and cheaper).
- test_northern_vs_southern_extents_differ_only_in_northing — DUP — the pyproj pin fixes every extent exactly.
- test_canonicalize_zone["33N", "15S", "01N"] — PARAM — identity cases; keep "1N", "60S", "33n", " 7s ".
- test_canonicalize_zone_rejects_malformed["", "33", "33NN", "abc", "-5N"] — PARAM — one regex rejects them; keep "N33", "0N", "61N", "33X" (range and hemisphere branches).

## tests/unit/storage/test_plan_update_script.py — KEEP — tests 2, LOC 64, cpu-s 0.1. `scripts/update/plan_update.py` is the operator entry point in docs/updates.md and these test its mask check, which no library test covers.

## tests/unit/config/test_config_inference.py — TRIM — tests 65→47, LOC 440→~360, cpu-s 0→0
- test_normalize_obs_checkpoints_sorts, _deduplicates, _filters_nonpositive — DUP — test_inference_config_num_obs_checkpoints_coerced feeds (16, 8, 8, 0) and asserts all three at once.
- test_normalize_obs_checkpoints_coerces_floats — PEDANTIC — float input is a type error mypy rejects.
- test_normalize_obs_checkpoints_empty_input_raises — PARAM — same raise as test_normalize_obs_checkpoints_empty_after_filter_raises.
- test_inference_config_empty_num_obs_checkpoints_raises — DUP — same `_normalize_obs_checkpoints` raise through the config.
- test_inference_config_default_norm_source_is_aws — DUP — test_an_explicitly_empty_norm_source_is_refused_not_defaulted asserts None → "aws".
- test_inference_config_invalid_norm_source_raises — DUP — same survivor ("" raises Invalid norm_source).
- test_inference_config_valid_s1_orbits — PEDANTIC — echoes the input.
- test_inference_config_default_model_version_is_v11 — PEDANTIC — a default constant.
- test_allow_s2_only_defaults_off — PEDANTIC — a default constant.
- test_the_newest_field_is_the_last_field — PEDANTIC — tripwire on field order; making the dataclass `kw_only=True` would enforce it structurally (an API decision for the owner).
- test_prefetch_depth_follows_vcpus_within_its_bounds[1-4-2, 0.6-8-4, 0.1-8-2] — PARAM — keep floor (1,1), mid (1,8), cap (1,64), share (0.5,8).
- TestBatchSizeForGpu::test_scales_only_below_the_tuned_card[TUNED_GPU_GIB, 24.0] — PARAM — keep 44.7 (tuned), 79.0 (bigger), 22.06 (A10G).
- TestBatchSizeForGpu::test_the_unfitted_batch_does_not_hold_it — PEDANTIC — arithmetic on module constants.

## tests/unit/config/test_fault_injection.py — TRIM — tests 36→27, LOC 401→~320, cpu-s 1.2→1.0
Production reach (grep src/): live, not test-only. `FaultInjection` is a parameter (default None) of flows fill_zone_year and fill_zones_sequential, which call `.arm()`; `ArmedFault` is consumed at storage/shard_writer.py, runners/zone_fill.py, runners/sequential_fill.py, inference/assembly.py; `hard_exit_after_flush` is also the wedged-drain exit in fill_zones_sequential (a non-drill production path). Keep the module's gate tests.
- TestArming::test_arming_is_refused_anywhere_but_a_drill_deployment[/yield-embeddings/ray/, /tessera/ray/, "", /some/deep/path/ray/] — PARAM — prefix parsing is pinned by test_the_deployment_is_read_off_the_injected_control_plane_prefix; keep prod, None, /global-tessera-dev/ (wrong shape). Drop its message asserts (PEDANTIC).
- TestArming::test_arming_announces_the_run_as_a_drill — PEDANTIC — log wording.
- TestWithholdingSupply::test_the_hold_announces_its_start_its_progress_and_its_end — PEDANTIC — log wording.
- TestWithholdingSupply::test_a_source_that_exhausts_before_any_hand_over_says_the_fault_did_not_fire — PEDANTIC — log wording; the None return is covered by test_a_fault_armed_for_the_other_kind_withholds_nothing.
- TestTheRequestItself::test_the_hold_is_bounded_at_both_ends[-1] — PARAM — 0 covers the lower bound.
- TestTheSharedHardExit::test_the_drill_still_goes_through_it — PEDANTIC — source-text grep; TestTheGuaranteeIsStructural::test_the_package_hard_exits_in_exactly_one_place already forces every hard exit through the shared helper.
Notes: TestTheGuaranteeIsStructural's two AST scans of all of src/ are architecture tests; MOVE to tests/architecture/. TestTheSharedHardExit re-imports `logging`, `os` and `fault_injection` inline in every test (already importable at module top).

## tests/unit/config/test_time_window.py — TRIM — tests 23→17, LOC 342→~230, cpu-s 0.2→0.2
- TestParseTimeWindow::test_december_window — DUP — test_various_end_months[calendar-year] asserts the same bounds.
- TestToDateRange::test_february_non_leap_year — PARAM — keep the leap case (calendar-aware is the point).
- TestToDateRange::test_31_day_end_month — PARAM — test_basic_range already checks a month-end.
- TestCheckTimeWindowCoverage::test_both_orbits_checks_all_three_stores — PARAM — parametrize with test_ascending_orbit_checks_two_stores over (orbit, expected stores).
- TestCheckTimeWindowCoverage::test_skip_coverage_check_still_rejects_empty_store — PARAM — parametrize test_empty_store_raises over skip_coverage_check.
- TestCheckTimeWindowCoverage::test_an_assessed_window_still_excuses_absent_months_when_data_exists — DUP — test_a_month_whose_dates_were_all_lost_is_excused_like_any_other is the same store plus an inert attr, and asserts more.
Notes: `check_time_window_coverage` lives in inference/data_loading.py; MOVE TestCheckTimeWindowCoverage into tests/unit/inference/test_assessed_window_coverage.py (same function; check that file for overlap with the assessed-window cases before moving). Six tests repeat an inline `import tessera_embeddings.inference.data_loading as dl` and a hand-written `_open_store`; let `_mock_open_store` take a group builder and reuse it (~60 LOC). `_assert_chronological` adds nothing over first/last/len on a generated range.

## tests/unit/config/test_global_store_uri_override.py — TRIM — tests 17→12, LOC 205→~150, cpu-s 0→0
- test_an_override_is_returned_verbatim — DUP — test_an_override_survives_serialisation asserts the same equality after a round trip.
- TestTheRegistrySitsBesideWhicheverStoreIsInUse::test_the_registry_is_never_inside_the_store — DUP — the exact-equality tests for derived, overridden and trailing-slash cases each pin a sibling path.
- ...::test_registry_parts_follow_the_store_not_our_outputs — DUP — test_a_named_store_gets_a_registry_beside_that_same_store asserts a part lands under its store's registry; the override case is test_an_overridden_store_takes_its_registry_with_it.
- ...::test_the_registry_root_cannot_be_asked_for_without_naming_a_store — PEDANTIC — asserts a TypeError for a missing positional arg (mypy's job).
- ...::test_registry_parts_are_partitioned_for_a_parquet_dataset — PEDANTIC — substring check; its docstring is also stale (parts are Parquet now, test_registry_master writes them with pq.write_table).

## tests/unit/config/test_inference_code_identity.py — TRIM — tests 15→13, LOC 192→~165, cpu-s 2.0→~0.5
- test_the_closure_is_reached_transitively_not_listed — DUP — test_content_determining_modules_are_covered[storage/time_axis.py] proves a transitively reached module is covered.
- test_the_digest_is_stable_across_calls — PEDANTIC — f()==f() and a prefix.
Slow cause: `_covered()` re-walks the whole import closure of the real src tree (AST parse of every reached file) on each of ~10 calls, and the ingest closure is rebuilt per param in test_targeted_updates_move_neither_fingerprint. Fix: module-scoped fixtures `covered` and `ingest_closure` (or `functools.cache` on the helpers).
Notes: test_a_relative_re_export_does_not_stop_the_closure and test_the_closure_holds_only_files_that_really_exist build the identical miniature package; share a fixture. test_the_real_ingest_fingerprint_covers_the_collection_settings re-imports `_MOSAIC_CONTENT_SOURCES` inline (already imported at the top).

## tests/unit/config/test_environment.py — TRIM — tests 13→9, LOC 157→~120, cpu-s 0→0
- TestGdalEnvironment::test_sets_all_expected_vars — PEDANTIC — re-lists ten constants from the source; the setdefault semantics are what matter.
- TestGdalEnvironment::test_respects_existing_values[GDAL_HTTP_MAX_RETRY, GDAL_CACHEMAX] — PARAM — one setdefault loop; keep GDAL_NUM_THREADS.
- TestPkgLoggerSetup::test_respects_src_log_level["WARNING"] — PARAM — "error" covers case-folding, "DEBUG" the other direction.
Keep the handler/propagation tests: they guard the doubled-CloudWatch-log regression.

## tests/unit/config/test_store_layout.py — TRIM — tests 13→10, LOC 140→~115, cpu-s 0.2→0.2
- test_single_embeddings_are_sharded_full_band — DUP — test_single_matches_global_geometry + test_global_embeddings_geometry imply it.
- test_global_scales_is_sharded_pcodec — PEDANTIC — re-states layout constants; test_global_store's `layout_departures(g) == []` and test_sharded_roundtrip_and_fill exercise the real codec.
- test_obs_vars_resolve_and_unknown_raises — PEDANTIC — dtype constant and a KeyError on an unknown dict key.

## tests/unit/config/test_ingest_worker_floor.py — TRIM — tests 6→4, LOC 55→~40, cpu-s 0
- test_default_is_a_fixed_size_fleet — DUP — test_the_floor_follows_each_leg_not_the_configured_maximum runs on the default `min_workers=None`.
- test_none_does_not_break_the_bounds_check — DUP — every default `IngestSettings()` construction exercises it.

## tests/unit/config/test_config_assembly.py — TRIM — tests 4→2, LOC 42→~22, cpu-s 0
- test_the_default_pool_is_sixteen_and_the_campaign_asks_for_thirty_two — PEDANTIC — two constants; it also makes a config test import the Prefect flow module `run_global_campaign` at collection.
- test_invalid_max_workers_raises — PARAM — parametrize with test_invalid_chunks_per_worker_raises.

## tests/unit/config/test_mosaic_layout.py — KEEP — 4 tests, 17 LOC.

## tests/unit/test_imports.py — DELETE — tests 9→0, LOC 94→0, cpu-s 0→0
- all 9 — DUP — every listed module is imported by its own unit test file at collection, so a circular import or missing dependency already fails collection; the hand-kept list also goes stale ("Update this file whenever a new public module is added").

## tests/unit/test_public_api.py — TRIM — tests 4→1, LOC 43→~12, cpu-s 0 (corrected 2026-11-05: was DELETE)
- test_public_api_doc_lists_every_exported_name — PEDANTIC — a docs test; owner ruling is not to test docs.
- KEEP test_every_exported_name_resolves. Ruff F822 does not cover it: F822 is skipped in `__init__.py`, and names resolved through the package's module-level `__getattr__` are invisible to it.
- test_all_contains_no_private_names, test_init_does_not_import_private_names — PEDANTIC — naming conventions on one file.

## tests/unit/test_prefect_layer.py — DELETE — tests 4→0, LOC 78→0, cpu-s 0.4→0
- test_prefect_imports_confined_to_orchestration_subtree — HISTORY — .importlinter contract "Prefect only in orchestration.prefect" enforces it in CI (architecture.yml).
- test_inference_orchestration_helpers_no_prefect — HISTORY — same contract (and the layers contract) covers that file.
- test_flow_modules_import, test_task_modules_import — DUP — every flow and task module is imported by its tests under tests/unit/flows and orchestration/.

## tests/unit/test_properties.py — DELETE — tests 8→0, LOC 192→0, cpu-s 1.6 local (CI far more)→0
- test_sliding_window_returns_one_pair_per_job, test_sliding_window_respects_concurrency_cap — DEAD — `sliding_window_submit` (orchestration/concurrency.py) has no caller in src, scripts or yield-embeddings, and is not in `__all__`/docs/public-api.md. Delete the module and tests/unit/orchestration/runners/test_orchestration_concurrency.py with it.
- test_store_for_is_deterministic — PEDANTIC — f()==f() on string formatting.
- test_store_for_distinguishes_kinds, test_store_for_path_shapes — DUP — test_global_store_uri_override::test_an_override_leaves_every_other_path_alone pins store_for's shapes; Hypothesis over an f-string adds nothing.
- test_roi_manifest_round_trip, test_ingest_manifest_round_trip — DUP — test_store_manifest::test_from_dict_round_trip and the identity/hash tests there.
- test_embedding_manifest_round_trip — DUP — test_store_manifest::test_embedding_manifest_from_dict_coerces_list_to_tuple (kept for this reason).
Slow cause: test_sliding_window_respects_concurrency_cap sleeps 5 ms per job, up to 20 jobs at concurrency 1, per Hypothesis example; the CI profile runs 200 examples (tests/conftest.py), so up to ~20 s per CI leg versus ~1.5 s locally.

## tests/architecture/test_quality_ratchet.py — KEEP — 8 tests, 74 LOC, 0.2 cpu-s. Guards the CI ratchet script, which has no other test.

## tests/architecture/test_cassette_safety.py — TRIM — tests 2→1, LOC 73→~60, cpu-s 0
- test_cassettes_directory_exists — PEDANTIC — asserts a directory and a README exist (a docs/layout check).

## Cross-file
- Totals for this slice: tests 937→568 (126 of the drop is test_zone_grid's 120-id parametrize collapsing to one loop), LOC ~10,200→~7,230, cpu-s ~153→~47.
- Biggest suite-wide lever (measured here, applies beyond this slice): `shard_writer.run_forked` always opens a spawn-context ProcessPoolExecutor, even for one payload, so every unit test that calls `write_year_shards` pays a fresh interpreter + package import (~1.5-2 s). Grep tests/unit/assembly and runners/test_zone_fill for `write_year_shards`/`run_forked`; a conftest fixture that swaps the pool for an in-process executor (for tests not about the fork machinery) is the cheap fix. The `write_year_shards` docstring ("in-process when 1") is stale.
- Per-test seeding of GLOBAL-layout stores recurs in test_published_store, test_fix_published_store_spatial_transform, test_global_store and test_campaign; add one session-scoped seeded template to tests/unit/storage/conftest.py and `shutil.copytree` it per mutating test.
- Dead src the cuts expose (~650 LOC): storage/region_writes.py + `write_region`/`_write_region`; `rollback_commits`; `set_s3_config`/`_s3_config_override`/`S3Config`; `CampaignStatus.pending`/`years_fully_complete`; orchestration/concurrency.py `sliding_window_submit` (its other test file, tests/unit/orchestration/runners/test_orchestration_concurrency.py, goes too); `sample_live_pixels` moves into its benchmark script.
- Shared dead list corrections: `daily_times` (storage/time_axis.py) and `resolve_region` (storage/zarr_store.py) are imported by yield-embeddings src; grep ../yield-embeddings/src before deleting anything on that list.
- Merges: test_write_days_windows_batched → test_write_day_windows; TestSavedManifestPreload (test_published_store) → test_global_store; TestCheckTimeWindowCoverage (config/test_time_window) → inference/test_assessed_window_coverage; fault_injection's AST scans → tests/architecture/; the four commit_year_attrs tests in test_campaign → assembly/test_shard_writer.
- Script tests: rebuild_registry_measurements re-asserts registry_master verdicts (layer duplication) and trims to its CLI plumbing; fix_published_store_spatial_transform is a one-shot whose fix is now structural (DELETE once the owner confirms the store was repaired); plan_update stays.
- Architecture tests already enforced elsewhere: Prefect confinement (.importlinter), delete that pytest copy. `__all__` names are NOT covered by ruff F822 (see test_public_api above); keep test_every_exported_name_resolves.
- Hidden CI cost: Hypothesis "ci" profile (200 examples) multiplies test_properties' 5 ms sleeps to ~20 s per CI leg; deleting the file removes it.
- Inline-import cleanup: test_open_or_create, test_write_day_windows, test_write_days_windows_batched, test_fault_injection (TestTheSharedHardExit), test_inference_code_identity, test_time_window (six `import ... data_loading as dl`).
