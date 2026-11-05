# Report: flows/, providers/, integration/, parity/

## tests/unit/orchestration/flows/test_run_global_campaign.py — TRIM — tests 118→94, LOC 1987→~1450, cpu-s 30→~6
- TestCampaignDefaults::test_ten_clusters_share_sixty_ingest_slots — PEDANTIC — pins signature defaults (10, 60) as literals.
- TestCampaignDefaults::test_the_actor_count_reaches_the_quota_without_exceeding_the_assembly_ceiling — PEDANTIC — pins default 250 literal.
- TestCampaignDefaults::test_the_pause_gate_is_published_as_running_not_as_a_cap — DUP — test_every_gate_is_published_to_the_server asserts inference gate == 1.
- TestCampaignDefaults::test_a_start_clears_a_pause_somebody_left_behind — DUP — same single write already asserted by test_every_gate_is_published_to_the_server.
- test_default_run_rejects_unseeded_work_zone — DUP — TestCampaignDefaults::test_a_rejected_invocation_never_touches_the_shared_gates (same stub; add `arun == []` there).
- TestStagingRunId::test_every_input_that_decides_the_embeddings_moves_it[optical_min_obs(30)] — PARAM — 25 already moves it; ids collide too.
- TestTheChainedFillGetsTheNarrowedStagingIdentity::test_an_inference_change_does — DUP — last assert of test_force_staging_reuse_survives_an_inference_change.
- TestTheChainedFillGetsTheNarrowedStagingIdentity::test_both_escape_hatches_reach_it — DUP — test_force_staging_reuse_survives… and test_force_staging_restage_forces_a_fresh_prefix.
- TestAPinnedStagingIdentity…::test_it_survives_a_changed_tarball — DUP — test_it_survives_an_inference_change: same verbatim-pin property.
- TestAPinnedStagingIdentity…::test_an_empty_pin_derives_exactly_as_before — DUP — default is "", so test_the_chained_child_is_handed_the_narrowed_identity covers it.
- TestValidationDeploymentForwarding::test_an_unset_validator_is_omitted…[some-branch] — PARAM — branch plays no part in omission; keep [None].
- TestIngestBoundAcrossClusters::test_the_default_split_is_five_zones_per_cluster — PARAM — second exact division (docstring stale: default is 10/60); keep exact/remainder/floor as one parametrize.
- test_branch_routes_fill_when_ingest_disabled — DUP — test_branch_routes_every_derived_ref; ingest flag does not touch routing.
- test_children_go_untagged_outside_a_flow_run — IMPROBABLE — only the `.fn` test path has no run id.
- TestFailedZonesAreRetried::test_a_failed_zone_is_re_dispatched — DUP — test_a_retry_targets_only_what_is_still_missing asserts both rounds.
- TestFailedZonesAreRetried::test_a_campaign_that_lands_everything_still_carries_the_key — PEDANTIC — key presence.
- test_overlap_years_dispatches_every_year_in_one_batch — DUP — test_overlapping_years_is_now_the_default (overlap is the default).
- test_the_dev_overlay_folds_the_tarball_etag_into_the_staging_identity — MOCK — asserts the stubbed ETag passes through; inline import.
- test_every_cluster_opens_on_one_of_the_densest_zones — DUP — test_cluster_balance::test_every_cluster_opens_on_one_of_the_densest_zones + test_every_cluster_runs_dense_to_sparse (real counts).
- test_chained_fills_shard_by_live_tiles — DUP — LPT in test_cluster_balance; look_ahead division in TestIngestBoundAcrossClusters.
- TestImmediateRefill::test_the_settling_delay_is_derived_from_the_cancellation_budget — PEDANTIC — constant equality.
- TestImmediateRefill::test_one_generation_per_slot_per_round — DUP — TestTheExtraAttemptIsBoundedAcrossRounds bounds attempts more strictly.
- TestImmediateRefill::test_a_round_that_is_already_over_is_not_even_planned_for — MOCK — counts stub calls (cost filter); outcome covered by test_the_last_cluster_to_settle_is_left_to_the_round.
- TestTheDispatchDecisionIsAtomic::test_a_round_that_ends_while_the_refill_is_planned_does_not_get_one — DUP — test_a_planner_is_dropped_once_no_dispatch_is_left_to_outrun (same sibling-settles-during-wait setup).
Notes:
- SLOW because the `wired` fixture leaves `mod.inference_code_identity` real: every dispatch re-hashes the import closure via `source_identity` (~80 ms/call, measured), and multi-round/multi-cluster tests dispatch 3-8 times. Fix: in `wired`, `monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-test")`; the five tests that care already override it. Saves ~24 cpu-s.
- Fold test_rejects_zero_parallel_ingest and TestFailedZonesAreRetried::test_zero_attempts_is_rejected into the `test_a_doomed_invocation_touches_no_shared_limit` parametrize (−20 LOC, test count unchanged); merge the two TestUncertaintyIsNotPermission tests into one parametrize; fold the four plain forwarding tests (s3_region, ingest=False, inference_start_delay, allow_s2_only) into one parametrized "forwarded to the fill" test.
- Not a thin shell: this flow is 1,960 LOC of real scheduling logic with no runner counterpart, so flow-level testing is the right layer. Layer overlap is only with test_cluster_balance (partition).
- immediate_refill (default off, no deployment enables it) still carries 16 tests after cuts; acceptable since campaign-plan.md says it is turned on at next restart.

## tests/unit/orchestration/flows/test_fill_zones_sequential_flow.py — TRIM — tests 92→76, LOC 1732→~1400, cpu-s 2.8→~2.0
- test_session_orbit_is_the_request — DUP — test_stream_contract_wired asserts session_s1_orbit == "both".
- test_model_guard_runs_before_ray — DUP — test_model_mismatch_prevents_cluster proves the guard precedes ray_up.
- test_live_cells_ordered_by_clamped_fleet_desc — DUP — test_cells_are_ordered_by_true_tile_count…; add one small zone there to keep the clamp assert.
- test_duplicate_and_lowercase_zones_are_canonicalized — DUP — test_an_exact_duplicate_cell_is_deduped_but_a_repeated_zone_is_not (uses "33n").
- test_the_same_zone_may_appear_for_several_years — DUP — same exact_duplicate test asserts live == 2 and both years.
- test_no_ingest_means_no_wait_before_the_cluster — DUP — test_ingest_false_skips_priming fails if the adapter is even built.
- test_ingest_true_wires_the_deployment_adapter — DUP — test_the_ingest_adapter_can_run_every_cell_the_flow_primes reads the real adapter's executor.
- TestPerCellValidation::test_nothing_is_dispatched_when_no_validator_is_configured — MOCK — asserts the stub got deployment=None; the decline is the helper's (test_cell_validation_dispatch).
- test_ingest_child_tag_derivation — PEDANTIC — tag string format; test_cancel_child_ingests_hook_sweeps_live_runs_by_tag asserts the same tag.
- test_adapter_wait_polls_child_run_to_completion — DUP — test_poll_retries_transient_errors_then_completes covers poll-to-terminal plus deregistration.
- test_adapter_refuses_dispatch_after_shutdown — DUP — test_a_dispatch_queued_while_shutdown_begins_creates_no_run is the real race.
- test_wait_first_does_not_abort_while_a_first_wave_cell_can_still_land — DUP — verbatim part (b) of test_the_priming_abort_quorum_is_the_first_wave.
- test_wait_first_raises_rather_than_booting_gpus_when_every_cell_failed — DUP — quorum part (a) + test_the_all_failed_report…; move its `__cause__` assert into (a).
- test_a_failure_while_priming_still_cancels_the_ingests_it_started[hold_for_backlog-RuntimeError] — PARAM — keep wait_first-RuntimeError and hold_for_backlog-TerminationSignal.
- TestWedgedDrainEndsTheProcess::test_the_flow_ends_the_process_only_after_its_teardown — PEDANTIC — greps the flow's source text for statement order.
Notes:
- Not slow. test_adapter_wait_honors_the_runner_stop_event takes 1.1 s (0.3 s Timer + shutdown confirm); cut the Timer to 0.05.
- Half the file (~800 LOC) tests `_DeploymentCellInputs`, a real threaded adapter living in the flow module; move those tests to their own file (test_deployment_cell_inputs.py). No runner duplicates them.
- Fold test_a_store_declaring_no_rule_leaves_the_config_unconstrained into test_the_chained_fill_applies_the_rule_its_store_advertises as a [25, None] parametrize; fold test_cluster_gets_idle_timeout_and_code_params into test_campaign_pins_reach_ray_cluster.
- test_every_year_is_validated_before_any_ingest_is_dispatched asserts `wired.get("ingest_dispatches")`, a key the fixture never sets (vacuous); keep the ray_kwargs assert, drop that line.
- test_the_helper_exits_through_the_one_sanctioned_site…: drop the `== 75` constant and source-grep asserts (PEDANTIC); keep the call assertion.

## tests/unit/orchestration/flows/test_ingest_zone_year.py — TRIM — tests 87→62, LOC 1756→~1250, cpu-s 0.9→0.7
- test_perf_report_uri_none_by_default — PEDANTIC — default value None.
- test_markerless_partial_mosaic_is_cleared — DUP — test_a_rootless_store_is_resumed_rather_than_wedging_every_retry (same resolve-after-dispatch shape); name is also stale.
- test_a_completed_sibling_store_survives_the_other_sensors_failure — HISTORY — asserts `not hasattr(mod, "delete_prefix")`; no delete path exists.
- test_partial_window_marker_does_not_satisfy_strict_run — DUP — test_a_complete_store_under_a_different_fingerprint_raises (any fingerprint field mismatch).
- TestChunkScaledWorkers::test_the_settings_value_is_a_cap_not_the_fleet_size — DUP — test_dense_zone_is_capped_by_settings.
- TestChildRunsAreCancellable::test_no_run_id_means_no_tag_rather_than_a_bogus_one — IMPROBABLE — only the `.fn` test path lacks a run id.
- test_local_mode_does_not_require_the_aws_provider — PEDANTIC — greps source line order.
- TestLegRetryClassification::test_transient_failures_are_retried[503, 403, ExpiredToken RasterioIOError] — DUP — taxonomy covered in tests/unit/ingest/test_read_failure_verdict.py.
- TestLegRetryClassification::test_transient_failures_are_retried[ThrottlingException, KilledWorker, TimeoutError] — PARAM — no marker matches; test_an_unrecognised_failure_is_retried covers default polarity. Keep token-expired, warp, Crashed, campaign-index-404.
- TestLegRetryClassification::test_deterministic_failures_are_not_retried[IReadBlock] — DUP — test_read_failure_verdict; keep the 404 id as the reuse proof.
- TestLegFailureDetail::test_a_missing_state_message_still_yields_a_detail — PEDANTIC — message text.
- TestTheLegStaggerDecorrelatesTheColdStart::test_the_two_legs_of_one_cell_do_not_share_an_offset — DUP — the spread test's >95%-unique assertion includes same-cell legs.
- test_max_leg_attempts_defaults_to_retrying — PEDANTIC — default constant.
- test_the_wall_clock_bound_is_on_by_default_and_is_a_real_duration — PEDANTIC — default constant.
- TestLegsRetryIndependently::test_a_terminal_failure_stops_a_siblings_remaining_attempts — DUP — test_a_terminal_leg_failure_is_not_forgotten_when_a_sibling_retries (one dispatch each).
- TestCatalogueRefusalSpendsTheBudgetDifferently::test_the_stopping_log_line_names_the_request_and_what_to_do — PEDANTIC — log wording.
- TestCatalogueRefusalSpendsTheBudgetDifferently::test_a_failure_carrying_no_refusal_token_is_unaffected — DUP — TestAPermanentLegFailureSkipsTheLadder::test_an_unrecognised_failure_still_spends_its_attempts.
- TestWallClockBoundsTheDecisionToStartAnotherAttempt::test_the_refusal_log_names_elapsed_deadline_and_the_resume — PEDANTIC — log wording.
- TestWallClockBounds…::test_a_load_refusal_keeps_its_attempts_inside_the_deadline — DUP — TestCatalogueRefusal…::test_a_repeated_load_refusal_keeps_every_attempt.
- TestAPermanentLegFailureSkipsTheLadder::test_only_the_words_cross_the_deployment_boundary — PEDANTIC — tests Prefect's state serialisation, not this code.
Notes:
- Fast (0.9 s); the autouse `waited` fixture already removes real sleeps. Savings are LOC only.
- Replace TestLegsRetryIndependently::test_a_dirty_mosaic_prefix_is_terminal_on_the_first_failure with a CorruptedStoreError id in test_deterministic_failures_are_not_retried; test_a_non_monotonic_date_is_dispatched_once stays as the one flow-level "permanent = one dispatch" test.
- Merge test_s3_region_threaded_through_metadata_opens and test_child_ingests_receive_the_configured_region into one test (same run, two assert groups).
- ~12 tests repeat the same two-line `zone_has_live_tiles`/`_probe_marker` stub; move it into a `live` fixture layered on `wired` (≈ −40 LOC). `_dispatch_counting`, `_dispatch_with_clock` and `_s2_leg_failing_every_attempt` are three near-identical fake-arun builders; collapse to one (≈ −50 LOC).

## tests/unit/providers/test_provider_aws_ray.py — TRIM — tests 74→50, LOC 1379→~950, cpu-s 13.7→~3
- test_resolve_ami_id_reads_the_ssm_pointer — MOCK — reads back the value moto was just given; no decision.
- test_cleanup_ray_tempfiles_handles_missing_path — PEDANTIC — no-op on None.
- TestRayDashboardSsmCommand::test_region_is_printed_so_a_differing_default_does_not_bite — PEDANTIC — log-line formatting.
- test_the_head_start_command_carries_both_halves — PEDANTIC — set relations between constant dicts.
- TestGpuWorkerLadderParsing::test_refuses_a_malformed_value[":16", "x", "1.5", ","] — PARAM — keep no-separator, no-count, negative, empty.
- TestGpuWorkerLadderApplication::test_absent_key_leaves_the_node_types_byte_identical — MOCK — loads the template twice and compares; never calls the code. test_resolve_ray_config_without_the_ladder_key_keeps_the_templates_node_types is the real guard.
- TestGpuWorkerLadderApplication::test_is_authoritative_not_additive — DUP — test_applies_counts_to_the_named_rungs asserts every unnamed rung is 0.
- TestGpuWorkerLadderApplication::test_a_closed_spot_rung_adds_nothing — DUP — test_raises_the_cluster_ceiling_to_fit_the_ladder (same arithmetic, spot=0).
- test_resolve_ray_config_refuses_a_malformed_ladder_before_writing_anything — DUP — parsing refusals + test_resolve_ray_config_reads_the_ladder_from_ssm cover parse and SSM plumbing.
- TestRayNodeTypePreference::test_every_g6e_size_carries_the_same_card — PEDANTIC — asserts facts of the static EC2 JSON fixture, no code runs.
- TestRayNodeTypePreference::test_each_card_carries_the_vram_the_catalogue_states[all 3] — PEDANTIC — fixture facts.
- TestRayNodeTypePreference::test_the_l40s_carries_exactly_twice_the_vram_of_the_24gb_cards — PEDANTIC — fixture arithmetic.
- TestRayNodeTypePreference::test_vcpu_per_gpu_ranks_the_candidate_sizes — PEDANTIC — fixture facts.
- TestRayNodeTypePreference::test_host_ram_per_gpu_clears_the_actor_budget_on_every_rung_we_ship — PEDANTIC — reads the fixture, not the template; TestL4Rungs::test_g6_xlarge_is_deliberately_not_offered guards the template.
- TestL4Rungs::test_the_2xlarge_of_every_card_family_is_the_ram_matched_shape — PEDANTIC — fixture facts.
- TestL4Rungs::test_an_uncapped_pair_of_fallback_cards_goes_entirely_to_the_l4 — HISTORY — docstring: policy superseded by fleet_mix/GPU_RUNGS ordering.
- TestL4Rungs::test_the_a10g_rung_alone_takes_the_whole_fallback_fleet — DUP — test_max_workers_zero_makes_a_rung_unreachable (one open rung takes all).
- TestL4Rungs::test_a_ladder_can_name_both_fallback_cards — DUP — test_applies_counts_to_the_named_rungs.
Notes:
- SLOW because every `_resolve_ray_config` test is `@mock_aws` and calls `_seed_ssm_and_vpc()` (VPC, 3 subnets, key pair, 7 SSM params) afresh: 0.5–1.8 s each, ~11 tests ≈ 10 cpu-s. Fix: a module-scoped `seeded_aws` fixture (`with mock_aws(): yield _seed_ssm_and_vpc()`); the two ladder tests delete their `gpu-worker-ladder` param in teardown. Also fold the two project-tag tests and pins_ami_id into one parametrized resolve test.
- test_ray_actually_reads_the_interval_from_the_environment spawns a Python subprocess (1 s) to probe Ray's own constant; move it to tests/integration beside test_ec2_launch_pacing. test_the_pacing_env_reaches_the_client_ray_builds_for_launches is the same kind of third-party canary; keep one of the two in unit.
- Trim test_g6_xlarge_is_deliberately_not_offered to its last assert (the fixture-value asserts are PEDANTIC).
- launch_pacing is off by default and only reachable via campaign/fill flags; 7 unit tests for it remain after cuts.

## tests/unit/providers/test_provider_aws_dask.py — TRIM — tests 65→44, LOC 980→~760, cpu-s 0.2→0.2
- TestRetryablePredicate::test_known_transient_errors_retry[all 7] — MOCK — parametrized over the src tuple, so it compares the tuple with itself (the file says so); real messages are tested below.
- TestRetryablePredicate::test_unrelated_runtime_error_does_not_retry — DUP — TestProvisioningThrottles::test_misconfiguration_still_fails_fast.
- TestProvisioningThrottles::test_misconfiguration_still_fails_fast[InvalidSubnetID, InvalidParameterException] — PARAM — keep AccessDenied and CannotPullContainerError.
- TestRetryPolicyWiring::test_retries_until_success — MOCK — builds its own tenacity Retrying in the test; tests tenacity, not the provider's policy.
- TestRetryPolicyWiring::test_gives_up_after_max_attempts_and_reraises_original — MOCK — same.
- TestRetryPolicyWiring::test_non_transient_fails_immediately — MOCK — same.
- TestSchedulerResourceLogger::test_line_reports_all_signals — DUP — test_emitted_line_parses_with_the_profiling_parser round-trips every field.
- TestSchedulerResourceLogger::test_memory_percent_uses_limit — DUP — the parse test asserts mem% against the limit.
- TestSchedulerResourceLogger::test_probe_before_start_is_a_noop — IMPROBABLE — the scheduler always calls start() before the PeriodicCallback fires.
- TestSchedulerResourceLogger::test_sampler_falls_back_to_main_thread_before_start — IMPROBABLE — docstring: the path exists for tests.
- test_ingest_settings_perf_report_uri_defaults_none — PEDANTIC — default constant.
- TestStopEcsTasksByTag::test_unparseable_arn_leaves_region_to_boto — IMPROBABLE — ECS_CLUSTER_ARN is CDK-injected, always a full ARN.
- TestStableDaskTaskDefinitions::test_only_one_set_pins_neither[worker only] — PARAM — mirror of [scheduler only].
- TestDaskContainerEnvironment::test_task_logs_still_ship_to_the_orchestrator_api — PEDANTIC — asserts absence of a setting nobody sets.
Notes:
- Fast. The decision-bearing tests (real AWS messages, region from ARN, tag sweep pagination, report containment, both-or-neither task defs) all stay.
- `_FakeScheduler`/`_FakeWorker`/`_FakeMemoryState` stay used; `Retrying`/`wait_fixed`/`stop_after_attempt` imports go with TestRetryPolicyWiring.

## tests/unit/orchestration/flows/test_dask_lifecycle.py — TRIM (move one) — tests 5→2 in unit (+1 in integration), LOC 79→~40, cpu-s 17.4→~0.1
- test_run_on_cluster_logs_onto_the_calling_flow_run — MOVE to tests/integration — not a cut: it is the only proof run_on_cluster attaches worker task runs + logs to the calling flow run (6 src callers).
- test_hook_without_a_run_id_is_a_noop — IMPROBABLE — Prefect always passes a flow_run carrying an id to terminal hooks.
- test_resource_tags_shape — PEDANTIC — dict literal / None passthrough.
Notes:
- WHY 17 s: `prefect_test_harness()` boots a real Prefect API server (a uvicorn subprocess on a temp SQLite DB, with migrations) per call, plus a `LocalCluster`, plus `APILogWorker.drain_all(timeout=5)`, then tears the server down. Nearly all of it is server start/stop, not the assertion. It is a Prefect-server integration test by nature.
- Fix: move it to tests/integration (that workflow already runs only when src/ changes). If it must stay in unit, reuse the session-scoped harness pattern from tests/parity/conftest.py so the server is booted at most once per session.

## tests/unit/orchestration/flows/test_fleet_gate.py — TRIM — tests 33→25, LOC 491→~420, cpu-s 0→0
- test_no_other_status_is_a_hold[401, 404, 500] — PARAM — keep 403 (client) and 503 (server).
- test_any_positive_limit_means_run[8, 61] — PARAM — keep [1], the value a campaign writes.
- test_a_full_gate_is_not_a_hold_here — IMPROBABLE — docstring: Prefect consumes 423 before this code sees it.
- test_entering_and_leaving_a_pause_are_both_logged — PEDANTIC — log wording.
- test_a_reading_is_cached_for_its_ttl — DUP — test_the_pause_is_noticed_within_a_ttl shows both the cached reading and its expiry.
Notes:
- Well built: real httpx/Prefect exceptions, no network. Drop the "global-concurrency-limit update" wording assert from test_a_hold_announces_itself_once_and_then_periodically (keep the count of 3).
- Inline `import threading` in test_the_gate_is_thread_safe duplicates the module import; remove.

## tests/unit/orchestration/flows/test_fill_zone_year_flow.py — TRIM — tests 21→17, LOC 371→~310, cpu-s 0→0
- test_allow_s2_only_reaches_inference_config[False] — PARAM — [True] proves the flag is threaded.
- TestPerCellValidation::test_the_summary_reaches_the_caller_unchanged — MOCK — returns the stubbed fill summary.
- TestTheMinimumDepthRuleComesFromTheStore::test_the_flow_takes_no_minimum_depth_parameter — PEDANTIC — signature introspection, and it inspects `mod.fill_zone_year` (the imported runner), not the flow.
- test_the_depth_rule_is_resolved_even_when_no_cluster_is_needed — PEDANTIC — splits the module source on "if needs_cluster:"; its monkeypatch is never exercised. Replace with a 6-line behavioural check on the retag path (config.optical_min_obs reaches the stubbed fill_zone_year) if the guarantee is wanted.
Notes:
- Merge the two `_optical_min_obs_from_store` reader tests into one parametrize ({} → None, {optical_min_obs: 25} → 25).
- Flow-vs-runner layering is clean here: the runner behaviour is stubbed and only gate ordering and wiring are asserted.

## tests/unit/orchestration/flows/test_seed_global_store.py — TRIM — tests 15→14, LOC 349→~260, cpu-s 0→0
- test_seed_stamps_the_model[v1.1] — PARAM — v1.1 is the default asserted by three other tests; keep [v2-large].
Notes:
- `_fully_seeded` and `_partially_seeded` are the same 25-line wiring differing only in how many zones are present; make one helper with `seeded_zones=len(ZONES)` default (−25 LOC). test_a_fully_seeded_store_still_validates_the_requested_identity re-inlines the same wiring a third time; route it through the helper and override `check_root_identity` (−30 LOC).
- Merge test_the_minimum_depth_rule_is_not_defaulted… and test_an_explicit_minimum_depth_rule_reaches_the_seeder into one [None, 25] parametrize.
- Every test guards a real seed decision (create vs reraise, rootless, write-once identity, landed-data refusal); no layer duplication with test_global_store.py beyond what the docstrings name.

## tests/unit/orchestration/flows/test_run_id_model_version.py — TRIM + MOVE — tests 19→11, LOC 243→~140, cpu-s 0→0
- test_fresh_run_ids_are_unique — PEDANTIC — uuid uniqueness.
- test_error_names_both_versions_so_the_fix_is_obvious — PEDANTIC — message text; the two refusal tests already match on "staged by v…".
- test_every_minting_site_prefixes_v2_and_leaves_v11_alone — DUP — test_a_matching_run_id_and_a_fresh_run_both_pass + runners/test_plain_runner.py::test_the_run_id_prefix_helper_is_public (prefix values).
- test_the_plain_runner_mints_an_id_its_own_guard_accepts — MOCK — rebuilds the minting expression inside the test, so it tests its own reconstruction.
- test_an_empty_run_id_shares_one_staging_prefix_across_models — PEDANTIC — documents the motivation by asserting a "//" path string.
- test_the_public_runner_refuses_an_empty_run_id_for_either_model[3 of 4 ids] — PARAM — the runner check is model-independent; keep ["", v1.1].
Notes:
- Only the first 7 tests touch the flow's `_resolve_run_id`; the rest test `config.inference.assert_run_id_matches_model` and `inference.runner`. Move those to tests/unit/config/ (or inference/) and fold the remaining `_resolve_run_id` tests into test_s2_only_flow_wiring.py, which tests the same function on the other axis — one file, `test_resolve_run_id.py`.

## tests/unit/orchestration/flows/test_s2_only_flow_wiring.py — TRIM + MERGE INTO test_resolve_run_id.py (with the above) — tests 18→15, LOC 245→~210, cpu-s 0→0
- test_fresh_run_id_is_bare_uuid_by_default — DUP — test_run_id_model_version::test_fresh_v11_run_keeps_the_historical_bare_uuid (identical call and asserts).
- test_full_pipeline_allow_s2_only_defaults_off — PEDANTIC — default value.
- test_full_pipeline_model_version_defaults_to_v11 — PEDANTIC — default value.

## tests/unit/orchestration/flows/test_flow_ray_teardown.py — TRIM — tests 11→9, LOC 211→~160, cpu-s 0→0
- test_cluster_name_is_deterministic_and_template_based — DUP — test_lifecycle_hook_derives_cluster_name_in_fresh_process asserts the same derived name.
- test_both_gpu_flows_terminate_retired_actors_instances — PEDANTIC — greps module source for two identifiers; test_fill_zone_year_flow::test_ray_path_wires_actor_terminator is the behavioural check.
Notes:
- Collapse the three `test_*_registers_teardown_on_cancel_and_crash` tests into one parametrize over the flow objects (−15 LOC).
- test_ray_owning_flows_do_not_import_the_aws_provider_at_module_scope is an import rule; the repo already runs import-linter, so express it as a `forbidden` contract there and delete the AST test (HISTORY once the contract lands).

## tests/unit/orchestration/flows/test_export_zone_rois_flow.py — KEEP — tests 10→10, LOC 149→149, cpu-s 0.4→0.4 (CONTEXT lists export_zone_rois as src-unreferenced, but it is a registered Prefect flow entry point documented in ingest/README.md and prefect/README.md; deployments live in the private repo, so not DEAD)

## tests/unit/orchestration/flows/test_cell_validation_dispatch.py — TRIM — tests 11→10, LOC 132→~125, cpu-s 0→0
- TestWhatIsDispatched::test_no_tag_outside_a_flow_run — IMPROBABLE — only `.fn` test calls lack a flow-run id (same pattern cut in three other files).

## tests/unit/orchestration/flows/test_pipeline_dates_width_gate.py — KEEP — tests 4→4, LOC 34→34, cpu-s 0→0

## tests/unit/orchestration/flows/test_flow_overrides.py — TRIM — tests 4→2, LOC 42→~25, cpu-s 0→0
- test_an_unset_parameter_is_omitted — DUP — test_set_and_unset_parameters_are_separated_in_one_call drops None.
- test_zero_is_a_choice_and_survives — DUP — the mixed-case test keeps `kept=0`.

## tests/unit/orchestration/flows/conftest.py — KEEP (18 LOC autouse `_store_holds_v11`; still used by three flow files)

## tests/unit/providers/test_published_store_credentials.py — TRIM — tests 32→25, LOC 528→~420, cpu-s 0→0
- TestTheExpiryWePromiseIcechunk::test_a_naive_expiry_is_not_compared_against_an_aware_one — IMPROBABLE — botocore always yields an aware expiry (the tzutc test below shows the real shape).
- TestTheAssumedRoleProviderItself::test_building_the_callback_makes_no_network_call — DUP — TestIcechunkCanActuallyUseTheCallback::test_the_callback_survives_the_pickling… builds the provider with no network.
- TestTheAssumedRoleProviderItself::test_repeated_calls_do_not_re_assume — DUP — fold the 5-call loop into test_the_callback_assumes_and_reports_a_capped_expiry (identical `_Sts`/`_Session` fakes, −25 LOC).
- TestTheWiringReachesTheStore::test_an_unconfigured_deployment_is_untouched — DUP — TestWhichCredentialAUriGets::test_with_nothing_configured_everything_uses_the_task_role (zone_fill delegates to the same selector).
- TestTheWiringReachesTheStore::test_the_registry_write_is_given_the_stores_credential — PEDANTIC — greps ZarrWriter source for a call string.
- TestIcechunkCredentialVisibility::test_a_workers_first_line_says_it_is_probably_a_change — PEDANTIC — log wording.
- TestIcechunkCredentialVisibility::test_both_providers_are_instrumented — PEDANTIC — greps two functions' source.
Notes:
- `cred` and `creds_mod` import the same module twice; use one name.
- Drop the second ("documents the contrast") assert in test_the_interval_lands_inside_botocores_mandatory_refresh_window — it pins a constant, not a decision.
- What stays guards real decisions: destination-based credential choice (incl. bucket-name prefix trap), expiry cap, pickling + tzinfo icechunk requirements, log ladder and lock.

## tests/unit/providers/test_fleet_mix.py — TRIM — tests 50→44, LOC 462→~420, cpu-s 0.3→0.3
- TestTheBoundsAreLoadBearing::test_a_draining_tail_does_not_buy_a_probe_it_cannot_use — DUP — TestTheCampaignShape::test_the_shape[draining queue] has identical inputs and expected asks.
- TestTheCampaignShape::test_the_live_fleet_stays_inside_the_budget_from_every_state[live0, live1, live3] — PARAM — test_the_shape already checks the budget for {} and {L40S:101}-like states; keep {L40S:51,A10G:79} (the regression) and {A10G:105}.
- TestOneFallbackAtATime::test_the_campaign_refuses_two_before_any_ingest — PEDANTIC — greps the campaign module text for a message; behaviour belongs in test_run_global_campaign's doomed-invocation parametrize.
- TestTheRunnerLifecycle::test_every_session_that_gets_a_terminator_also_gets_the_publisher — HISTORY — AST check written for two call sites; one remains, and losing `on_fleet_demand` there fails the lifecycle tests' flow-level counterpart in test_fill_zones_sequential_flow.
Notes:
- Merge test_meaningless_inputs_are_refused and test_a_negative_demand_is_refused into one parametrize.
- `_log()` and `_config()` use inline imports; hoist to module scope (CLAUDE.md: no inline imports).
- TestTheRunnerLifecycle tests inference.runner, not the provider; it is correctly patched (ray.kill) and fast, but belongs beside tests/unit/inference/test_scheduling.py.

## tests/unit/providers/test_gpu_capacity_fallback.py — TRIM — tests 24→17, LOC 296→~190, cpu-s 0.3→0.2
- TestTheCampaignRestartConfiguration::test_it_yields_101_l40s_and_105_a10g — DUP — test_fleet_mix::TestTheCampaignShape::test_the_ceilings_come_out_at_101_and_105 (same ladder and budget).
- TestTheCampaignRestartConfiguration::test_the_realistic_fleet_fits_the_account_quota — PEDANTIC — arithmetic on test-local constants (39 L40S observed).
- TestTheCampaignRestartConfiguration::test_the_l40s_ceiling_leaves_room_to_grow — PEDANTIC — 101 > a test constant.
- TestTheCampaignRestartConfiguration::test_a_fully_supplied_fleet_would_exceed_the_quota — PEDANTIC — docstring: "stated rather than guarded".
- TestVcpuBudget::test_each_pure_fleet_lands_exactly_on_the_budget — DUP — test_each_rung_is_ceilinged_at_what_the_budget_affords_it (250×4 = 125×8 = 1000).
- TestVcpuBudget::test_single_gpu_rungs_are_unaffected_by_the_per_instance_rule — DUP — same arithmetic at 840.
- TestVcpuBudget::test_absent_budget_keeps_the_production_ceiling — DUP — TestApplyGpuFallback::test_opens_the_named_card_at_the_production_ceiling (identical asserts).
Notes:
- DEAD helper: `_node_types()` is never called (its scorer tests were deleted, per the module docstring); remove it with `ACTOR_BUNDLE`, `EC2_TYPES`, `np_mod`, `json`, `patch` imports (−20 LOC).
- Merge test_a_budget_too_small_to_seat_a_node_is_refused and test_a_budget_that_seats_production_but_not_the_fallback_is_refused into one [1, 4] parametrize.
- Layer overlap with test_provider_aws_ray TestGpuWorkerLadderApplication: none; the ladder is tested there, the fallback here.

## tests/parity/test_ingest_s2_roi_date_loop_parity.py — TRIM (move one test) — tests 4→3 in parity (+1 in unit), LOC 187→~175, cpu-s 12.5→~0 in the default run
- test_batch_dates_validation_rejects_bad_combinations — MOVE to tests/unit/ingest/test_s2_date_pipeline.py — the only test of `batch_dates must be >= 1` (s2_roi.py:411); not a cut.
Notes:
- THE ANOMALY EXPLAINED: this test carries no `parity` marker, so root `addopts` (`-m 'not integration and not parity and not slow and not gpu'`) does not deselect it, and `testpaths = ["tests"]` collects tests/parity in any bare `pytest` run. tests/parity/conftest.py defines `_isolated_prefect_runtime` as a SESSION-scoped AUTOUSE fixture that enters `prefect_test_harness()` (boots a Prefect API server on temp SQLite), so this 6-line validation check pays the whole server boot: 12.5 s of setup, ~0 s of body. It is also never run in CI: unit CI runs `pytest tests/unit`, and parity-fast selects `-m "parity and not slow"`, which deselects it. Moving it to tests/unit fixes both.
- The three [batched / pipelined / batched-and-pipelined] parity runs are local (LocalCluster + local icechunk, synthetic band load): no network.

## tests/parity/test_ingest_s1_roi_parity.py — DELETE — tests 1→0, LOC 169→0, cpu-s 0→0
- test_s1_roi_parity — HISTORY — `xfail(strict=False, raises=Exception)` since the CMR-native query change (issue #45) and `skipif` without EDL credentials: it can only skip or xfail, so it guards nothing. Needs network + EDL creds.
Notes:
- Delete with it: tests/fixtures/stac_cassettes/test_s1_roi_parity.yaml (220 KB); update scripts/build/record_stac_cassettes.py and scripts/diagnostic/probe_edl_bearer.py references. Re-add when issue #45 is fixed.

## tests/parity/test_ingest_s2_roi_parity.py — KEEP — tests 1→1, LOC 98→98 (NETWORK: STAC from cassette, but COG pixel reads go live to Earth Search's public S3; fails if unreachable)

## tests/parity/test_generate_roi_parity.py — KEEP — tests 1→1, LOC 85→85 (local only; the ADR 001 "flow == domain" check for the thinnest flow)

## tests/parity/adapter_template/test_your_adapter_parity.py — DELETE — tests 1→0, LOC 49→0
- test_your_adapter_parity — DEAD — permanently `skip`-marked template; collected and skipped every parity run. Move the snippet into tests/parity/adapter_template/README.md (or the parity README) as a code block.

## tests/parity/conftest.py + helpers.py — KEEP (341 LOC). `assert_zarr_equivalent`/`stage_quickstart_roi` stay used by the remaining parity files.

## tests/integration/test_stac_query_cassette.py — DELETE — tests 1→0, LOC 69→0
- test_s2_stac_search_against_denver — MOCK — calls only pystac-client against a recording of pystac-client; no repo code runs. Cassette infrastructure is exercised by test_ingest_s2_roi_parity and test_baseline_producer_cassette.
Notes:
- Its cassette test_s2_stac_search_against_denver.yaml is byte-identical (same md5) to test_s2_roi_parity.yaml; delete it (585 KB).

## tests/integration/test_baseline_producer_cassette.py — TRIM — tests 6→4, LOC 284→~250 (no network: cassette replay)
- TestRecordedProducerSplit::test_raw_items_report_old_baselines — PEDANTIC — asserts a fact about the recorded archive window, not code ("evidence, not a census").
- TestRecordedProducerSplit::test_provenance_records_the_real_baselines — PEDANTIC — only checks the result is one of the declared values.
Notes:
- The five TestRecordedProducerSplit tests each carry their OWN cassette file, and all five are byte-identical (101,682 bytes, same md5). Use one class-level cassette name (`@pytest.mark.vcr("TestRecordedProducerSplit.yaml")`) and delete four (and two more after the cuts) files, −400 KB.

## tests/integration/test_read_failure_cause_over_dask.py — TRIM — tests 8→3, LOC 122→~110 (no network: spawned LocalCluster)
- test_restoring_the_cause_does_not_widen_what_gets_skipped[TIFFReadEncodedTile, 503, 429, 401, Connection reset] — DUP — the verdict per message is unit-tested in tests/unit/ingest/test_read_failure_verdict.py; the cluster only has to prove the cause survives for one skip and one non-skip (keep ZIPDecode and ASF_DENIED).
Notes:
- SLOW: `spawned_cluster` is function-scoped and spawns two worker processes per test (8 clusters). After the cut, make it module-scoped for the parametrized test; the A/B test needs a fresh (un-rescued) cluster, so give it its own.

## tests/integration/test_ec2_launch_pacing.py (+ _launch_pacing_sim.py 252 LOC) — TRIM / GATE — tests 4→2, LOC 450→~410 (no AWS: a local EC2 stand-in in subprocesses)
- test_the_paced_arm_settles_near_the_buckets_refill_rate — PEDANTIC — loose two-sided bounds on a laptop-dependent rate; the floor half duplicates test_a_paced_fleet_keeps_asking_while_the_bucket_is_shut.
- test_pacing_makes_fewer_launch_requests_for_more_fleet — DUP — test_pacing_costs_less_quota_per_instance_placed carries the claim; add its `paced.placed >= control.placed` assert there.
Notes:
- The module fixture `arms` runs three arms of "minutes of subprocesses" each, in the integration workflow on every src change, for `launch_pacing` which is off by default. That breaks tests/integration/README.md's own "> 2 minutes" rule. Gate it behind an env var like test_published_store_access (or record its result in context_docs and delete), and move unit's test_ray_actually_reads_the_interval_from_the_environment beside it.

## tests/integration/test_published_store_access.py — KEEP — tests 8→8, LOC 182→182 (NETWORK: live anonymous S3 read of the published store; opt-in via TESSERA_TEST_PUBLISHED_STORE=1, which no CI job sets, so it never runs in CI)

## tests/integration/test_ray_joined_gpu_accounting.py — KEEP — tests 1→1, LOC 119→119 (local Ray, no network; pins Ray's cluster_resources semantics)

## tests/integration/test_dask_scheduler_plugin.py — KEEP — tests 1→1, LOC 56→56 (LocalCluster, no network)

## tests/unit/orchestration/flows/test_cluster_balance.py — TRIM — tests 35→18, LOC 213→~185, cpu-s 0→0
- test_every_cluster_opens_on_one_of_the_densest_zones[2, 4, 6, 12, 16, 24] — PARAM — keep [1, 8, 40, 120] (trivial, near default, knee, degenerate).
- test_every_cluster_runs_dense_to_sparse[2, 4, 6, 12, 16, 24] — PARAM — same surviving ids.
- test_clusters_finish_together[1, 4, 6, 12] — PARAM — n=1 is trivially balanced; keep [2, 8, 16].
- test_report — DUP — a print utility whose only assertion repeats test_every_cluster_opens_on_one_of_the_densest_zones; turn it into a script under scripts/ if the tables are wanted.
Notes:
- This is the real-coverage home of the LPT partition; two synthetic-weight duplicates in test_run_global_campaign are cut against it. Docstrings say the default is 8 clusters; it is 10 (DEFAULT_CLUSTERS is read from the flow, so the assertions are right, the prose is stale).

## Cross-file
- Slow time in this slice is three fixtures, not test count: `wired` in test_run_global_campaign leaves `inference_code_identity` real (~80 ms import-closure hash per dispatch, ≈24 cpu-s); test_dask_lifecycle boots a Prefect API server via `prefect_test_harness` (17 s); test_provider_aws_ray re-seeds moto VPC/SSM per test (≈10 cpu-s). The parity anomaly is the same server boot via an autouse session fixture (12.5 s, local runs only).
- The "terminal hook registered on cancel AND crash" assertion appears in six files (run_global_campaign, fill_zones_sequential_flow, ingest_zone_year, dask_lifecycle, flow_ray_teardown ×3). One parametrized test over (flow, hook) pairs in test_flow_ray_teardown.py (renamed test_terminal_hooks.py) replaces seven tests.
- "No run id → no tag" (IMPROBABLE, `.fn`-only) was cut in four files; "default value is X" PEDANTIC tests were cut in five.
- Source-grep / AST tests (`inspect.getsource`, `read_text().split`) cut in six places; the one worth keeping as a rule (no module-scope `providers.aws` import in Ray flows) belongs in import-linter.
- Layer duplication, flow vs runner: small. The flows in this slice are NOT thin shells: run_global_campaign (1,960 LOC), ingest_zone_year (1,197) and fill_zones_sequential's `_DeploymentCellInputs` hold real logic with no runner counterpart, so flow-level tests are the right layer. Real duplicates found: LPT partition (campaign vs test_cluster_balance), retry taxonomy (ingest_zone_year vs tests/unit/ingest/test_read_failure_verdict.py), campaign restart ceilings (test_gpu_capacity_fallback vs test_fleet_mix), `_resolve_run_id` split across two flow files.
- Misplaced: test_run_id_model_version (mostly config/inference), TestTheRunnerLifecycle in test_fleet_mix (inference.runner), `_DeploymentCellInputs` tests (half of test_fill_zones_sequential_flow).
- Cassettes: 6 of 10 files under tests/fixtures/stac_cassettes are byte duplicates or orphaned by these cuts (≈1.2 MB).
- CI: tests/integration runs `-m integration`, so any unmarked integration test would never run in CI; all are currently marked. test_published_store_access and test_ingest_s1_roi_parity never run in CI (env/creds gated).
