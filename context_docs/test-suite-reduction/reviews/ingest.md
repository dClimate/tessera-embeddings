# Ingest slice report

## tests/unit/ingest/test_duplicate_granules.py — TRIM — tests 270→~150, LOC 2436→~1450, cpu-s 0.1→0.1
Key finding: the read-failure classifier (`classify_read_failure_in`, duplicates.py:723) is ONE ordered function; `is_unreadable_source` only returns True on a codec marker (`_UNREADABLE_MARKERS` = ZIPDecode, TIFFReadEncodedTile) or missing-object+reader marker. Every "X does not step down" test whose chain has NO codec marker passes even if X's exclusion is deleted. ~45 parametrize ids are vacuous this way. Replace them with ONE precedence param: `<marker>: ZIPDecode...` must not step down, over ~6 ids (ExpiredToken, InvalidAccessKeyId, AccessDenied, HTTP 503, HTTP 401, Connection refused) — extend TestANamedTransportFailureIsNotBadData::test_it_takes_precedence_over_a_bad_data_marker_in_the_same_chain, which is the only correctly built one.
- TestWhenToStepDown::test_a_bare_wrapper_is_not_a_decode_failure — PARAM — 3 ids → keep only the IReadBlock id (the one that used to match).
- TestWhenToStepDown::test_a_statusless_transport_failure_is_not_bad_data (5) — DUP — vacuous; TestANamedTransport precedence test covers the same 5 messages properly.
- TestWhenToStepDown::test_a_missing_source_object_means_step_down — PARAM — 3 ids → keep NoSuchKey + "specified key" (2).
- TestWhenToStepDown::test_a_missing_object_nobody_attributes_to_the_source_does_not_step_down — PARAM — 3 ids → keep IcechunkError + NoSuchKey.
- TestWhenToStepDown::test_a_credential_or_throttle_failure_must_not_step_down (4) — DUP — vacuous (no codec marker); move ExpiredToken/AccessDenied into the precedence param.
- TestWhenToStepDown::test_a_credential_failure_wrapped_in_the_decode_wrapper_still_does_not_step_down — DUP — vacuous: chain has no codec marker; becomes the ExpiredToken precedence id.
- TestWhenTheProviderRefused (whole class, 34 ids) — DUP — every id lacks a codec marker so all are vacuous; status ranges already covered positively by TestTheTwoVerdictsCannotDisagree.
  (includes test_the_warp_wrapper_is_looked_through... which asserts the same line twice; test_data_that_will_never_decode_is_still_claimed DUP of test_a_decode_failure_means_step_down; test_an_unrecognised_failure_belongs_to_neither DUP of test_an_unrelated_failure_does_not_step_down.)
- TestAMessageMayNotBeItsOwnCorroboration (whole class, 3) — DUP — credential ids vacuous (no codec marker); corrupt-data positive DUP of test_a_decode_failure_means_step_down.
- TestTheTwoVerdictsCannotDisagree::test_every_transient_status_is_a_refusal — PARAM — 8 ids → keep 503, 599, 408, 429.
- TestTheTwoVerdictsCannotDisagree::test_the_ranged_readers_wording_carries_a_status_too — PARAM — 4 ids → keep 403, 503.
- TestANamedTransportFailureIsNotBadData::test_it_takes_precedence... — PARAM — 5 transport ids → keep 3 (resolve host, refused, reset) plus the new precedence ids.
- TestRecognisingAFailureThatArrivedWithNoEvidence::test_it_gets_no_verdict_rather_than_the_wrong_one — DUP — TestTheTwoVerdicts::test_a_stripped_cause_earns_neither_verdict.
- TestKeyingACopyAcrossProviders::test_an_unreadable_identifier_names_no_image — PARAM — 2x5=10 ids → field x [None, truncated id] = 4.
- TestIdentifyingACopy::test_the_sequence_is_read_as_an_integer — DUP — TestASequenceIsMatchedNotCoerced::test_a_real_sequence_parses["12"].
- TestIdentifyingACopy::test_the_sequence_falls_back_to_the_id — DUP — TestASequence::test_an_absent_property_defers_to_the_id.
- TestIdentifyingACopy::test_an_unreadable_sequence_is_none — DUP — TestASequence::test_an_unreadable_sequence_with_no_id_sequence_is_none.
- TestASequenceIsMatchedNotCoerced::test_a_real_sequence_parses — PARAM — 4 ids → "12", 3 (int).
- TestASequenceIsMatchedNotCoerced::test_anything_else_defers_to_the_id — PARAM — 14 ids → 6 (truncating-float, infinity, negative, exponent, arabic-indic, bool).
- TestChoosingBetweenCopies::test_the_newest_copy_is_kept — DUP — test_catalogue_order_does_not_decide (same pair, both orders) + test_a_single_image_steps_exactly_as_before (alternates).
- TestTheFallbackLadder::test_the_label_names_the_copy_per_tile — PEDANTIC — log label formatting.
- TestADatastripSplitIsTwoImages::test_reprocessing_keeps_the_image — DUP — TestKeyingACopy::test_an_image_is_its_pass_and_datastrip_without_the_baseline.
- TestDistinctImagesOnOneDay (whole class, 3) — DUP — TestAgainstRealPystacItems (copies reduce, other image survives) + TestTheLadderStepsOneImageAtATime::test_stepping_down_keeps_the_other_image (asserts kept==[a_new,b_new]).
- TestInRegionPreference::test_the_remote_copy_stays_available_as_a_fallback — DUP — fold its one assert into test_equal_baseline_prefers_the_in_region_copy (identical fixture).
- TestInRegionPreference::test_below_the_threshold_the_correction_term_is_inert — DUP — identical fixture/assert to test_locality_never_outranks_a_newer_baseline (02.06 local vs 03.01 remote).
- TestInRegionPreference::test_owing_no_correction_no_longer_costs_a_reprocessing — DUP — TestASpareThatWillRefuse::test_a_spare_that_merely_owes_the_offset_is_now_offered (same 04.00 local / 05.00 remote pair).
- TestInRegionPreference::test_an_all_remote_date_still_selects_one — PARAM — same rule as test_sequence_still_decides_when_baseline_and_locality_tie; parametrize host.
- TestLocalityPredicate::test_bands_all_in_a_preferred_host_are_local — DUP — TestLocalityNeedsTheCompleteReadSet::test_the_complete_read_set_is_local.
- TestLocalityPredicate::test_the_remote_bucket_is_not_in_the_preferred_set — PEDANTIC — asserts a constant; the locality tests fail if it changes.
- TestAssetBucketParsing::test_lookalikes_are_not_the_preferred_bucket — PARAM — 4 ids → "-backup" s3 + attacker host.
- TestAMalformedHrefDoesNotAbortTheSelection::test_an_unparseable_href_reads_as_remote — PARAM — 3 ids → 1.
- TestUnknownBaselineSuspendsLocality::test_an_unknown_baseline_does_not_win_on_locality — DUP — test_a_usable_baseline_beats_a_higher_sequence_without_one (same rule, also asserts ladder exclusion).
- TestLocalityNeedsTheCompleteReadSet::test_a_read_key_missing_from_an_otherwise_local_item_is_not_local — DUP — test_one_local_band_is_not_a_local_item.
- TestLocalityNeedsTheCompleteReadSet::test_an_item_with_no_assets_is_incomplete — DUP — TestLocalityPredicate::test_an_item_exposing_none_of_the_read_bands_is_not_local.
- TestNonFiniteBaselinesAreUnknown::test_a_non_finite_baseline_reads_as_unknown — PARAM — 6 ids → "NaN", "inf".
- TestNonFiniteBaselinesAreUnknown::test_infinity_does_not_outrank_a_real_baseline — DUP — parse test + TestTheLadderKeepsGlobalPriority::test_an_unreadable_baseline_sorts_last.
- TestUnknownBaselineStaysScopedToItsImage::test_the_best_spare_overall_is_still_first — DUP — TestTheLadderKeepsGlobalPriority::test_every_position_is_in_descending_baseline_order.
- TestTheLadderKeepsGlobalPriorityPastItsHead::test_a_second_retry_does_not_skip_a_better_copy_in_another_image — DUP — test_every_position_is_in_descending_baseline_order (same property, more rungs).
- TestThePreferenceKeyHoldsItsInvariants::test_locality_never_outranks_a_higher_baseline — DUP — TestInRegionPreference::test_locality_never_outranks_a_newer_baseline (same baselines).
- TestThePreferenceKeyHoldsItsInvariants::test_completeness_outranks_the_baseline — DUP — TestCompletenessOutranks...::test_a_complete_copy_beats_a_higher_sequence_incomplete_one.
- TestThePreferenceKeyHoldsItsInvariants::test_locality_does_not_participate_when_the_baseline_is_unreadable — DUP — BROKEN: both items are local=False, so locality is never varied; TestLocalityDoesNotDecideBetweenUnknownBaselines covers it.
- TestThePreferenceKeyHoldsItsInvariants::test_a_harmonised_copy_is_not_penalised_on_the_readable_term — DUP — TestTheLadderOnlyOffers::test_a_harmonised_spare_with_no_baseline_is_still_offered.
- TestThePreferenceKeyHoldsItsInvariants::test_an_incomplete_copy_loses_to_an_otherwise_identical_complete_one — DUP — TestLocalityNeedsTheCompleteReadSet::test_an_incomplete_copy_loses_an_otherwise_total_tie.
- TestCompletenessOutranksTheSequenceWhenABaselineIsUnreadable::test_sequence_still_decides_between_two_complete_copies — DUP — sequence tie-break covered by TestInRegionPreference::test_sequence_still_decides....
- TestAMixedProducerCopyIsRankedOnItsBaselineLikeAnyOther::test_below_the_threshold_the_baseline_still_decides — PARAM — parametrize test_a_mixed_producer_copy... over baseline.
- TestAMixedProducerCopyIsRankedOnItsBaselineLikeAnyOther::test_two_homogeneous_copies_are_still_ranked_on_baseline — DUP — baseline ordering asserted by a dozen tests.
- TestTheLadderOnlyOffersCopiesItCanRecoverWith::test_a_usable_spare_is_still_offered — DUP — TestASpareThatWillRefuse::test_a_usable_spare_is_still_offered.
- TestTheLadderOnlyOffersCopiesItCanRecoverWith::test_a_spare_with_an_unreadable_baseline_is_excluded — DUP — TestUnknownBaselineSuspendsLocality::test_a_usable_baseline_beats... (asserts alternates == {}).
- TestTheLadderOnlyOffersCopiesItCanRecoverWith::test_a_mixed_producer_spare_is_kept_below_the_threshold — PARAM — merge with ..._offered_at_every_baseline_now over baseline.
- TestTheDuplicateLogIsAnAuditTrail::test_it_states_the_actual_preference — PEDANTIC — pins log prose.
- TestTheDuplicateLogIsAnAuditTrail::test_it_states_one_ranking_because_there_is_only_one — PEDANTIC — same log prose again.
- TestTheAuditReportsWhatSelectionDidNot...::test_nothing_pruned_logs_nothing — DUP — TestTheDuplicateLogIsAnAuditTrail::test_no_duplicates_logs_nothing.
- TestTheLadderStepsNothingWhenTheFailedImageHasNoSpare::test_the_failed_image_is_stepped_when_it_does_have_a_spare — DUP — module-level test_the_ladder_steps_the_image_that_failed_not_the_best_ranked_spare.
- TestTheLadderStepsNothingWhenTheFailedImageHasNoSpare::test_unattributed_failure_still_steps_the_best_spare — DUP — module-level test_without_attribution_the_ladder_still_steps_the_best_ranked_spare.
- TestTheProducerTermIsInertWithoutAReadSet::test_the_newer_copy_still_wins — DUP — test_the_spare_stays_on_the_fallback_ladder asserts the same winner.
- TestTheProducerTermIsInertWithoutAReadSet::test_a_post_threshold_copy_does_not_read_as_refusing — DUP — TestASpareThatWillRefuse::test_without_the_collection_answer_the_spare_looks_harmless.
Notes: ~40% of the LOC is docstring narrative ("REVERSED", "Raised on PR #107", "USED TO STEP DOWN") — edit history the owner forbids in docs; cut each to one line (~400 LOC). Ten near-identical `_copy`/`_at`/`_item` helpers across classes collapse into `_copy_at` + `_with_assets` (~60 LOC). Three `_emit` caplog helpers → one module helper; TestTheDuplicateLogIsAnAuditTrail._emit has a dead `copies_label  # noqa: B018` line and inline imports of log_duplicate_selection (move to top). TestASpareThatWillRefuse::test_the_driver_supplies_it... inline-imports private s2_roi helpers (move import up). Class names are sentences; consider grouping into 5 classes (keying, ranking, ladder, audit log, failure classifier) — the classifier tests belong in a test_read_failure_verdict.py merge (see that file).

## tests/unit/ingest/test_stac_ingest.py — TRIM — tests 89→56, LOC 1364→~1000, cpu-s 0.8→0.5
- TestBaselineCorrection::test_no_corrected_pixel_can_wrap — DUP — test_a_dark_pixel_is_corrected... already feeds uint16 500; caller only corrects pixels > nodata (stac.py:833).
- TestBaselineCorrection::test_the_brightest_codes_lose_exactly_the_offset — PARAM — fold into test_subtracts_the_offset as one more row.
- TestBaselineCorrection::test_a_valid_dark_pixel_does_not_become_nodata — DUP — test_a_dark_pixel_is_corrected... asserts 999→1 (same floor).
- TestDateFiltering::test_filter_existing_dates_with_empty_items_returns_empty — PEDANTIC — empty in, empty out.
- TestBaselineExtraction::test_extract_baseline_parses_510_correctly — PARAM — same as the 400 case; extract_baselines test covers 510.
- TestProviderConfiguration::test_get_provider_config_returns_expected_provider (2) — PEDANTIC — echoes catalogue URL constants.
- TestProviderConfiguration::test_get_collection_config_returns_correct_bands (4) — PEDANTIC — asserts config.bands == the constant it was built from.
- TestProviderConfiguration::test_baseline_correction_flag_set_correctly — PARAM — 5 ids → ES S2 (True) + ES S1 (False); its inline comment is also stale (says S2 False, expects True).
- TestBuildStacQueryBboxFallback::test_uses_tile_property_when_available — DUP — TestSTACQueryBuilder[earth-search-sentinel-2-l2a].
- TestBuildStacQueryBboxFallback::test_tile_id_with_tile_property_unchanged — DUP — same.
- TestIngestTile::test_ingest_tile_applies_baseline_correction — DUP — test_ingest_tile_does_not_correct_extra_bands also requires a non-None driver (reads driver.md_parser).
- TestIngestTile::test_ingest_tile_applies_item_filter_fn — DEAD — no src/scripts caller passes item_filter_fn (grep `item_filter_fn=` outside stac.py: 0); also asserts almost nothing.
- TestIngestTile::test_ingest_tile_item_filter_fn_returns_none_when_empty — DEAD — same parameter.
- TestItemPruning::test_a_complete_match_still_prunes — DUP — test_drops_unread_assets_and_links (identical call and assert).
- TestItemPruning::test_item_without_assets_is_returned_unchanged — DUP — test_unrecognised_asset_names_leave_the_item_untouched (same no-match branch).
- test_the_duplicate_audit_log_names_the_area_it_selected_for — MOCK — stubs five internals to check a log label string.
- TestSplitQueryWindow::test_the_parts_cover_every_day_of_the_input — DUP — test_the_union_is_the_input_exactly... (stronger, on the wire form).
- TestSplitQueryWindow::test_a_seam_does_not_repeat_a_whole_day — DUP — union test + test_two_days_split_into_one_day_each.
- TestTheWindowTree...::test_every_window_is_walked_and_read_back_depth_first — PARAM — 6 worker ids → [1, 8].
- TestTheWindowTree...::test_the_read_back_order_is_identical_at_every_width — DUP — the param test asserts preorder == shape at each width (stronger).
- TestTheWindowTree...::test_two_siblings_failing_at_once_raise_the_earlier_one — PARAM — 4 ids → [1, 8].
- TestTheSortIsTotal...::test_a_tie_is_broken_by_id_rather_than_by_arrival_order — DUP — test_the_same_items_in_any_arrival_order_come_out_the_same.
- TestTheSortIsTotal...::test_the_cloud_ordering_still_wins_over_the_tie_breaker — DUP — same test asserts forward[:3] cloud order.
Notes: item_filter_fn is a dead parameter of query_stac_items/has_new_stac_dates/ingest_tile (src cleanup candidate). Window-tree tests sleep 0-6 ms per node over ~40 nodes x 10 runs; the PARAM cuts remove most of the 0.3 s. Eight copies of the "33UUP is UTM zone 33..." comment and the ingest_tile kwargs block → one `_ingest(**overrides)` helper (~120 LOC). Inline imports (odc.stac, pystac.Item, stac_mod) to the top.

## tests/unit/ingest/test_loader_failure_attribution.py — TRIM — tests 68→56, LOC 1137→~950, cpu-s 0.4→0.15
- TestNamingTheObject::test_the_label_caps_and_says_it_capped — PEDANTIC — log-label formatting ("+12 more", comma count).
- TestCapturingWhatGdalLogsButDoesNotRaise::test_only_a_refusal_is_recorded — PARAM — 5 ids → sidecar-404, no-response-at-all, range-past-the-end (the three with a status the classifier could misread).
- TestTheReasonGdalDidNotRaise::test_the_evidence_travels_with_the_exception — DUP — test_the_logged_refusal_decides_the_same_failure_the_other_way (same setup, superset of asserts).
- TestTheReasonGdalDidNotRaise::test_nothing_logged_changes_nothing — DUP — test_without_the_log_the_refusal_reads_as_bad_imagery (same no-log path).
- TestTheReasonGdalDidNotRaise::test_attached_evidence_can_never_cost_a_date[LOGGED_REFUSAL] — PARAM — DUP of ..._decides_the_same_failure...; keep the bare-503 id.
- TestMappingObjectsBackToCopies::test_no_hrefs_attributes_nothing — DUP — test_an_href_from_another_date_attributes_nothing (same empty-set answer).
- TestWhatTheLogSays::test_the_label_names_only_what_was_stepped — PEDANTIC — log label text.
- TestWhatTheLogSays::test_a_whole_date_label_is_capped — PEDANTIC — log label suffix.
- TestTheReasonDoesNotSurviveTheWorkerBoundary::test_the_verdict_is_undecidable — DUP — test_duplicate_granules TestRecognisingAFailure...::test_the_production_shape_is_recognised + test_the_cause_is_replaced_by_the_wrappers_repr.
- TestTheReasonDoesNotSurviveTheWorkerBoundary::test_the_bare_warp_wrapper_arrives_saying_nothing_at_all — DUP — test_duplicate_granules ...::test_the_bare_warp_wrapper_is_recognised.
- TestTheReasonSurvivesOnceTheReducerIsInstalled::test_the_gdal_error_number_survives_too — DUP — fold `args == (1, 4, CODEC_FAILURE)` into test_the_whole_chain_arrives.
Notes: TestCapturingWhatGdalLogs::test_a_slow_collection... sleeps 0.2 s and three tests sleep 0.05 s; monkeypatch `loader_failures.time.monotonic` instead (saves ~0.35 s). clear_local_refusals (test-only src def) is used only by the `capture_installed` fixture — legitimate test seam, keep. TestWhatTheLogSays::test_narrowing_alternates_is_an_intersection stays (the `only=` argument is not covered in test_duplicate_granules).

## tests/unit/ingest/test_baseline_correction_by_source.py — TRIM — tests 93→58, LOC 1066→~650, cpu-s 0.1→0.1
- TestPerItemExemption::test_a_raw_esa_item_below_the_threshold_reports_its_baseline — DUP — TestTheBaselineParserReportsUnknown::test_a_real_baseline_is_scaled_by_a_hundred.
- TestPerItemExemption::test_an_unreadable_baseline_reports_zero (3) — DUP — TestTheBaselineParser... param ids + test_a_declared_zero_is_distinguishable_from_an_absent_one.
- TestPerItemExemption::test_two_dates_from_different_producers_get_different_answers — DUP — the raw-over-threshold and harmonised-over-threshold tests above it, combined.
- TestProvenanceIsNotOverwritten::test_a_harmonised_item_still_reports_its_real_baseline — DUP — TestPerItemExemption::test_a_harmonised_item_over_the_threshold_is_left_alone asserts extract_baselines.
- TestHarmonisationPredicate::test_the_live_catalogue_serves_the_configured_alias_keys — DUP — identical body to test_a_harmonised_bucket_is_recognised.
- TestHarmonisationPredicate::test_the_raw_archive_is_not_in_the_harmonised_set — PEDANTIC — constant; behaviour tests fail if it changes.
- TestHarmonisationPredicate::test_an_unrecognised_bucket_is_not_harmonised — DUP — TestAnUnlistedBucketIsUndeterminedNotRaw::test_a_complete_set_from_an_unlisted_bucket_is_undetermined.
- TestHarmonisationPredicate::test_an_item_serving_bands_under_native_keys_is_undetermined — DUP — TestThePerItemCheck...::test_a_native_keyed_item_reads_as_undetermined_on_its_own.
- TestHarmonisationPredicate::test_bands_straddling_two_producers_are_not_harmonised — DUP — TestMixedNeedsBothKnownProducers::test_straddling_two_known_producers_is_still_mixed (same fixture; 4 copies of it exist).
- TestSclDoesNotDecideHarmonisation::test_scl_is_read_but_not_reflectance — PEDANTIC — constants; raw-scl behaviour test covers it.
- TestSclDoesNotDecideHarmonisation::test_a_raw_reflectance_band_still_makes_it_mixed — DUP — TestMixedNeeds...::test_straddling_two_known_producers_is_still_mixed.
- TestTheBaselineParserReportsUnknown::test_an_undeclared_or_nonsense_baseline_is_unknown — PARAM — 6 ids → None, "not-a-number", "Infinity".
- TestTheBaselineParserReportsUnknown::test_a_real_baseline_is_scaled_by_a_hundred — PARAM — 3 ids → "05.10", "00.01".
- TestTheBaselineParserReportsUnknown::test_an_infinite_baseline_does_not_raise_out_of_the_parser — DUP — the "Infinity" param id (same parser path).
- TestThePerItemCheckIsScoped...::test_the_planetary_computer_collection_does_not_use_the_per_item_check — PEDANTIC — config flag; wiring asserted by test_duplicate_granules TestASpareThatWillRefuse::test_the_driver_supplies_it....
- TestThePerItemCheckIsScoped...::test_earth_search_does_use_the_per_item_check — PEDANTIC — same.
- TestTheTwoBucketSetsAreIndependent::test_the_raw_archive_is_absent_from_both — PEDANTIC — constants.
- TestScalingCannotOverflowThePicker::test_a_value_that_overflows_when_scaled_reads_as_unknown — PARAM — 3 ids → "1e308".
- TestScalingCannotOverflowThePicker::test_the_batch_is_not_aborted_by_one_such_item — DUP — the param above runs the same parser; its assert (`in (510, 0)`) is near-vacuous.
- TestRankingUsesTheRequestedAssetsNotThePruningSet::test_an_explicit_extra_band_is_requested — DUP — test_the_requested_set_does_not_duplicate_a_configured_band (asserts "qa" requested).
- TestRankingUsesTheRequestedAssetsNotThePruningSet::test_a_copy_lacking_scl_is_not_penalised_when_scl_is_not_read — DUP — test_duplicate_granules TestRankingJudgesTheAssetsThisLoadWillRequest::test_locality_is_judged_over_the_requested_set_too.
- TestProvenanceRealignsWheneverSelectionPruned::test_the_fixture_really_yields_no_alternates — DUP — fold its two asserts into test_the_map_is_realigned_even_with_no_alternates.
- TestTheThemesThatGeneratedTheReviewStayClosed::test_there_is_one_definition_of_what_a_load_reads — DUP — TestPruningReaches...::test_the_read_set_rule_has_one_definition (both providers).
- TestTheThemes...::test_the_read_set_is_empty_where_the_names_are_not_the_asset_keys — DUP — test_duplicate_granules ...::test_the_driver_supplies_it_for_the_provider_that_needs_it.
- TestTheThemes...::test_scl_is_read_but_never_judged_for_harmonisation — PEDANTIC — constant set difference.
- TestTheThemes...::test_provenance_is_never_the_correction_input — DUP — TestPerItemExemption::test_a_harmonised_item_over_the_threshold_is_left_alone (identical asserts).
- TestPruningReaches...::test_the_gate_stops_at_collections_owed_no_offset — PEDANTIC — config flag; also test_stac_ingest flag param + leaves_a_collection_owed_no_offset_alone.
- TestAnUnlistedBucketIsUndeterminedNotRaw::test_the_named_raw_archive_is_still_raw — DUP — TestHarmonisationPredicate::test_raw_esa_is_not (3 identical copies).
- TestAnUnlistedBucketIsUndeterminedNotRaw::test_a_harmonised_producer_is_still_harmonised — DUP — TestHarmonisationPredicate::test_a_harmonised_bucket_is_recognised.
- TestRawRequiresEveryBandFromAKnownProducer::test_every_band_from_the_known_archive_is_raw — DUP — TestHarmonisationPredicate::test_raw_esa_is_not.
- TestRawRequiresEveryBandFromAKnownProducer::test_raw_plus_harmonised_is_still_mixed — DUP — TestMixedNeeds...::test_straddling_two_known_producers_is_still_mixed.
Notes: seven identical `xr.Dataset` + `fake_load` blocks → one `loaded_ids` fixture monkeypatching `_load_from_stac` (~70 LOC); five `_pair()` builders differing only in host/baseline → one `_pair(winner, loser, *, loser_first)` helper (~60 LOC). Classes are one-per-review-finding ("Raised on PR #107"); regroup into ~5 (harmonisation predicate, baseline parser, per-source decision, selection-at-load/provenance, read-set config). Baseline parsing is tested in THREE files (here, test_stac_ingest TestBaselineExtraction, test_duplicate_granules TestNonFiniteBaselinesAreUnknown): keep it here only. Inline imports of s2_roi privates → top of file.

## tests/unit/ingest/test_catalogue_refusal.py — TRIM — tests 57→47, LOC 959→~800, cpu-s 0.3→0.3
- TestClassification::test_the_two_refusals_take_different_paths — DUP — the two parametrized classification tests assert kind and repeat_is_deterministic per status.
- TestClassification::test_the_ladder_is_chosen_by_the_providers_own_flags — PEDANTIC — pins config flags and exact status sets; flag behaviour is test_only_the_provider_that_refuses_oversized_pages_skips_the_502_retry.
- TestTheFailingRequestIsNamed::test_a_refusal_before_any_search_is_named_as_the_catalogue_root — DUP — TestTheCatalogueRootKeeps...::test_a_root_502_that_never_clears_is_reported_as_page_zero (same page==0/"catalogue-root" asserts).
- TestRefusalLabels::test_the_label_says_whether_patience_was_already_spent — PEDANTIC — label wording.
- TestRefusalLabels::test_the_root_request_label_says_it_precedes_the_search — PEDANTIC — label wording.
- TestARefusalNoShorterWindow...::test_the_page_halves_to_a_floor_and_then_the_refusal_is_reported — DUP — test_the_floor_is_reached_by_halving... drives the identical call; fold the kind/page asserts in.
- TestARefusalNoShorterWindow...::test_a_backend_failure_is_not_re_asked_as_a_smaller_request — PARAM — [500, 504] → [504].
- TestARefusalNoShorterWindow...::test_a_stated_overload_on_the_first_page_is_not_answered_with_a_smaller_page — DUP — TestADeepRefusalIsRePartitioned::test_a_load_refusal_is_left_to_the_retry_ladder (503 propagates as LOAD); add the "items per page" assert there.
- TestARefusalThatKeepsArrivingCannotAmplify::test_it_gives_up_after_a_bounded_number_of_replacements — DUP — test_the_bound_does_not_grow_with_the_window runs the same month; add its isinstance assert there.
- test_the_window_walk_concurrency_stays_low — PEDANTIC — asserts a constant's range.
Notes: well-built file (real urllib3 ladder, no hand-written messages). Five copies of the `patch.object(stac, "StacApiIO") / patch.object(stac, "Client")` + `_query_stac_items(...bbox=(-3.0, 50.0, -2.0, 51.0))` block → one module helper (~50 LOC).

## tests/unit/ingest/test_s2_date_pipeline.py — TRIM — tests 53→36, LOC 882→~740, cpu-s 4.3→~2.6 (wall falls more: ~2.5 s of sleep)
- test_the_pipeline_line_is_emitted_once_per_written_date [both] — PEDANTIC — pins a log line kept for a finished serial-vs-pipelined A/B.
- test_the_pipeline_lines_fields_are_non_negative [both] — PEDANTIC — same log line's arithmetic.
- test_serial_mode_reports_nothing_hidden — PEDANTIC — same log line.
- test_stage_timings_total_stays_the_serial_equivalent_cost [both] — PEDANTIC — log-field arithmetic with a 0.15 s tolerance (timing-flaky).
- test_a_resume_offers_nothing_at_or_before_the_newest_held_date[pipelined] — PARAM — resume start is computed before the date loop; mode adds nothing. Keep [serial].
- test_a_store_already_past_the_window_is_a_no_op[pipelined] — PARAM — same.
- test_a_no_op_resume_still_repairs_the_assessed_window[pipelined] — PARAM — same.
- test_the_assessed_window_records_the_requested_start_not_the_resumed_one[pipelined] — PARAM — same.
- test_a_window_whose_end_precedes_its_start_is_refused[pipelined] — PARAM — validation runs before the loop.
- test_a_compact_iso_window_is_not_read_as_closed[pipelined] — PARAM — same.
- test_a_bound_that_is_not_a_date_is_refused[pipelined-*] (2) — PARAM — same.
- test_dense_dates_are_written_at_once_and_prepared_one_ahead[dense] — PARAM — sparse-then-dense reaches the same dense-run assertion.
- test_a_batch_holds_sparse_dates_up_to_its_chunk_budget[chunks0-None] — PARAM — same outcome as the budget=10 sparse case.
Notes: slow because every test drives the whole `ingest_s2_roi_reflectance` (run_ingest fixture, function scope) with `_FakeClient` computing the coverage gate on dask's local threaded scheduler per date, plus real `time.sleep(write_s)`: write_s=0.2 in the dense test (4-7 dates x 2 ids ≈ 2.2 s wall) and 0.1 in test_the_next_date_is_prepared... (0.6 s). Cheapest fixes: the PARAM cuts above (BOTH_MODES doubles 17 tests whose subject sits outside the loop); drop write_s to 0.03 (overlap is still observable — it's an ordering assertion); set `dask.config.set(scheduler="synchronous")` in the fixture to remove thread-pool overhead. Two inline `from tessera_embeddings.ingest import s2_roi` re-imports (already imported at top) and one inline tenacity import in the autouse fixture → top.

## tests/unit/ingest/test_s1_batch_lookahead.py — TRIM — tests 27→18, LOC 653→~470, cpu-s 1.3→0.6
- test_next_batch_is_prepared_during_the_current_batch_write — DUP — test_pipeline.py::test_prepare_overlaps_consume (same `pipelined`, depth=1).
- test_first_batch_stalls_for_its_whole_query_and_later_ones_do_not — DUP — test_pipeline.py::test_stall_measures_starvation (~0.75 s of sleep).
- test_prepare_exceptions_surface_when_the_batch_is_consumed — DUP — test_pipeline.py::test_prepare_exception_surfaces_in_order.
- test_per_date_narrowing_is_priced_like_the_run — PARAM — 2 ids → [overlapped-writes]; the constant is threaded through the same kwarg either way.
- test_a_store_already_past_the_window_is_a_no_op — DUP — test_a_no_op_resume_still_repairs_the_assessed_window makes the identical call and asserts more.
- test_an_unusable_batch_width_is_refused_whatever_the_store_holds — PARAM — 3 ids → ["a store already past the window"], the only one that reaches the new check.
- test_a_bound_that_is_not_a_date_is_refused — PARAM — 2 ids → ["2018-02-30"].
Notes: the resume rule itself (`solar_days.resume_window_start`, `validated_window`) is tested in test_solar_days; the S1 and S2 legs each re-test it through the full loop. Keep one wiring test per leg (no-op resume repairs assessed window + requested start recorded + compact ISO), as above. test_narrowing_off_writes_every_window repeats test_narrowing_writes_only... as its first half — drop that half. Six inline imports (numpy, xarray, s1_roi, datetime, SimpleNamespace, live_windows constants) → top.

## tests/unit/ingest/test_pipeline.py — TRIM — tests 10→8, LOC 236→~200, cpu-s 2.3→0.4 (it is wall-clock sleep, not CPU)
- test_depth_one_is_unchanged_and_default — DUP — test_prepare_overlaps_consume (default depth, order and overlap).
- test_single_item_stalls_for_its_whole_preparation — DUP — test_stall_measures_starvation (starved branch).
Notes: slow only because of real sleeps: test_stall_measures_starvation sleeps 1.2 s, test_prepare_overlaps_consume ~0.65 s. Overlap and stall are ORDERING properties; shrink PREPARE_S/CONSUME_S to 0.01/0.03 and the stall sleeps to 0.05 (keep a 3x ratio), saving ~1.8 s wall. This is the canonical home for `pipelined`; the three copies in test_s1_batch_lookahead go.

## tests/unit/ingest/test_boa_offset.py — TRIM — tests 36→27, LOC 566→~480, cpu-s 1.0→0.7
- TestSourceDecision::test_a_harmonised_source_needs_no_readable_baseline — DUP — test_a_harmonised_source_is_never_owed_the_offset loops over baseline None.
- TestSourceDecision::test_an_answer_naming_no_producer_cannot_be_passed_at_all — PEDANTIC — asserts a type alias's args; mypy enforces it.
- TestNothingCanSupplyAnUnsettledProducer::test_no_configured_collection_yields_an_unsettled_producer — DUP — test_the_collection_answer_is_settled_for_every_configuration is exhaustive over both fields.
- TestTheParserStampsEachSource::test_scl_is_never_stamped — DUP — TestTheOffsetIsRemovedPerImage::test_scl_is_untouched_by_a_load_that_corrects_everything_else (real load, stronger).
- TestTheOffsetIsRemovedPerImage::test_the_stored_dtype_is_the_one_the_store_expects — DUP — fold the dtype assert into test_scl_is_untouched... (identical single-tile load).
- TestTheOffsetIsRemovedPerImage::test_nodata_survives_the_correction — DUP — test_scl_is_untouched... already asserts {0, 500} on blue.
- TestNoExistingPixelChanges::test_a_day_that_used_to_load_is_unchanged — PARAM/HISTORY — 4 ids → raw-04.00 (threshold inclusive) + raw-02.06; it compares against a reference copy of the deleted date-wide corrector, a migration proof; the harmonised id repeats TestTheOffsetIsRemovedPerImage.
- TestTheDriverDoesNotDisturbTheRead::test_it_reads_through_odcs_own_reader — PEDANTIC — issubclass check; the real-load tests prove the reader runs.
Notes: cost is real `odc.stac.load` over tmp GeoTIFFs (~0.1 s each); the cuts above remove 4 such loads. source_decision is also exercised indirectly by the `_decisions` helper in test_baseline_correction_by_source — keep the decision table here and let that file test only harmonisation/selection. Inline imports (`typing`, `PROVIDERS` x4 — already imported at top as config.providers.PROVIDERS, `odc.loader`, `_BoaCorrectingReader`) → top.

## tests/unit/ingest/test_land_mask.py — TRIM — tests 42→36, LOC 518→~495, cpu-s 1.6→1.4
- test_parse_cell_name_rejects_malformed — PARAM — 4 ids → "red.tiff", "grid_1_2_3.tiff".
- test_parse_cell_name_rejects_invalid_coords — PARAM — 5 ids → nan, off-lattice (2.4), cell-edge (10.0); the two out-of-range ids hit the same bounds check.
- test_export_zone_roi_ocean_returns_none — DUP — test_export_zone_roi_removes_a_superseded_mask_when_a_zone_turns_all_ocean asserts the same None.
- test_validate_zone_roi_accepts_freshly_exported — DUP — the misplaced-land and missing-chunk tests both assert a fresh export validates to [] first.
Notes: validate_zone_roi / export_zone_rois have no src caller but are the documented export_zone_rois flow (prefect README) — live, keep. Cost is real icechunk/zarr writes via `make_coverage` per test (function scope, ~40 ms each); a module-scoped coverage fixture for the read-only 31N [(10,5),(11,5)] case would save ~0.3 s but the tests mutate the exported ROI, so only the coverage half can be shared.

## tests/unit/ingest/test_roi_mask_credential_expiry.py — MERGE INTO test_roi_mask.py (with the two below) — tests 6→3, LOC 329→~250, cpu-s 12.3→~2
- test_the_mask_still_reads_the_right_pixels — DUP — pixel totals asserted by the radar and second-compute tests; layout by test_roi_mask_construction::test_the_mask_is_the_array_that_was_written.
- test_a_credential_that_expires_after_the_graph_is_built_does_not_fail_the_write — DUP — test_a_second_compute_of_one_graph_re_resolves_again (build, expire, compute x3) + radar test; the write wrapper adds no credential path.
- test_the_optical_window_total_survives_a_credential_that_expired_after_graph_build — DUP — `_sum_over_windows` is a reduction over the same `read_roi_mask` graph the second-compute test reduces.
Notes: WHY SLOW: the `role_only_s3` fixture (function scope) configures `credential_process = {sys.executable} issue_role_credential.py` with a 2-second expiry, so EVERY credential resolution forks a fresh Python interpreter (~60-100 ms CPU each), and each test triggers dozens (graph build + one per dask-thread block read, re-resolved on every compute; the second-compute test does it 3x). Requests also go through a threaded HTTP proxy (`_FrontDoor`) in front of moto. Cheapest safe fix: write the issuer as a `/bin/sh` script (`printf '{"Version":1,"AccessKeyId":"AKIAROLE%s",...}' "$(date +%s%N)"`) — same unique-per-call keys, no interpreter start-up — plus the three DUP cuts. Expect ~12 s → ~1.5 s.

## tests/unit/ingest/test_roi_mask_construction.py — MERGE INTO test_roi_mask.py — tests 14→10, LOC 342→~300, cpu-s 7.1→~3.5
- test_the_mask_is_the_array_that_was_written — PARAM — 6 ids → ragged-both-axes, chunk-larger-than-array, single-row (exact-multiple and single-column add no new slice arithmetic; one-pixel-blocks builds 289 tasks for nothing).
- test_the_closure_survives_a_process_based_scheduler[dict] — PARAM — the provider-callable id is the stricter case (must arrive callable on the worker); keep it alone.
Notes: WHY SLOW: test_the_closure_survives_a_process_based_scheduler starts a `distributed.LocalCluster(n_workers=2, processes=True)` per param (~3 s each: two worker processes importing the package). Keeping one param halves it; `n_workers=1` halves it again (serialization is proven by one process). The AST source checks (test_every_caller_of_apply_roi_mask_supplies_the_mask, test_each_sensor_reads_through..., test_no_sensor_freezes...) are cheap structural guards — keep.

## tests/unit/ingest/test_roi_mask_credentials.py — MERGE INTO test_roi_mask.py — tests 13→4, LOC 154→~70, cpu-s 0.1→0.1
- test_a_dict_is_passed_through_unchanged — PEDANTIC — identity passthrough.
- test_none_stays_none — PEDANTIC — identity passthrough.
- test_a_provider_is_invoked — DUP — test_a_provider_is_invoked_on_every_call_never_memoised.
- test_the_mask_reader_resolves_a_provider — DUP — expiry tests call read_roi_mask with iam_s3_storage_options (a provider) and assert per-read resolution.
- test_the_mask_reader_still_reads_correctly_through_a_provider — DUP — construction process-scheduler test with the provider-callable form compares to the written truth.
- test_no_module_narrows_the_annotation_back_to_a_dict (4) — PEDANTIC — greps one literal annotation string; mypy rejects a provider passed to a dict-typed parameter.
Notes: survivors: test_a_provider_is_invoked_on_every_call_never_memoised, and the three TestTheCallersPassAProviderAndReResolve wiring checks (string-grep on tasks/ingest.py is brittle; prefer an AST check like its siblings).

## tests/unit/ingest/test_solar_days.py — TRIM (+ absorbs test_solar_day_grouping) — tests 84→~57, LOC 417→~470 after merge, cpu-s 0.4→0.3
- test_every_owned_solar_day_is_wholly_inside_the_queried_range — PARAM — 9 longitudes x 3 batch sizes = 27 ids → longitudes [99, -99, 0, 180, -180] x batch_days [1, 30] = 10.
- test_a_straddling_day_is_owned_by_exactly_one_batch_and_owned_whole — PARAM — 9 ids → [99, -99, 0, 180, -180].
- test_normalising_twice_is_normalising_once — PARAM — 11 ids → [99, 0, 180, -180, 179.9].
- test_owned_ranges_tile_the_window_exactly — PARAM — drop the "batch7" producer (4 ids); batch30 exercises the same fixed_day_ranges arithmetic.
- test_normalisation_is_idempotent — DUP — test_normalising_twice_is_normalising_once (parametrized, stricter: three passes).
- test_an_already_canonical_stamp_is_left_alone — DUP — test_a_real_noon_acquisition_still_gets_its_offset + test_normalising_twice... cover the noon-stamp mechanism.
Notes: TestResumeWindowStart is the canonical home of the resume rule; the S1/S2 leg files re-test it through their loops (see those sections).

## tests/unit/ingest/test_solar_day_grouping.py — MERGE INTO test_solar_days.py — tests 8→6, LOC 87→~60
- test_offset_matches_the_loaders_whole_hour_truncation — DUP — test_solar_days::test_the_dateline_keeps_its_true_twelve_hour_offset (keep its 14.9→0 truncation assert there).
- test_central_longitudes_are_unaffected — PARAM — 3 ids → [0.0].
Notes: survivors (far-east across UTC midnight, one UTC date holding two solar days, UTC fallback, within-group order) exercise `group_items_by_date` + normalisation — they fit under test_solar_days' "grouping" section.

## tests/unit/ingest/test_solar_day_labelling.py — DELETE — tests 5→0, LOC 107→0
- test_two_solar_days_can_share_a_utc_date_at_a_large_offset — MOCK — asserts the test file's own `_solar_day_of` helper, no src.
- test_the_label_must_not_come_from_the_cloudiest_item — MOCK — compares two computations both done in the test; no src call.
- test_every_item_in_a_group_yields_the_same_solar_day — DUP — test_solar_day_grouping::test_far_east_acquisitions_across_utc_midnight_are_one_solar_day.
- test_labels_are_monotonic_across_consecutive_solar_days — MOCK — test helper only.
- test_mid_longitude_zones_are_unaffected — MOCK — test helper only.
Notes: the real label is `solar_day_of` (src), covered by test_solar_days::test_deriving_a_date_from_an_unnormalised_item_raises and the ownership tests.

## tests/unit/ingest/test_solar_day_fusion.py — KEEP — tests 3, LOC 153, cpu-s 0.1 (real odc fuse over GeoTIFFs; the one test of first-wins fusion on pixels)

## tests/unit/ingest/test_s2_coverage_windows.py — MERGE INTO test_s2_coverage_gate.py — tests 6→4, LOC 136→~90
- test_the_equivalence_is_not_vacuous — MOCK — proves the test helper `_sum` sums only what it is given; no src property.
- test_multiple_bands_are_disjoint — DUP — merge_bands disjointness belongs to test_live_windows (which owns merge_bands); here it re-derives windows by hand with a zero price.
Notes: same subject as test_s2_coverage_gate (the cropped ratio of `_coverage_from_scl` / `_sum_over_windows`); one file "test_s2_coverage_gate.py" with a "denominator" and a "numerator" section.

## tests/unit/ingest/test_s2_coverage_gate.py — TRIM (+ absorbs test_s2_coverage_windows) — tests 9→7, LOC 155→~240 after merge
- test_threshold_boundaries[0.0] — PARAM — test_only_the_declared_invalid_classes_are_excluded already runs at 0.0; keep [100.0].
- test_no_windows_at_all_fails_rather_than_dividing_by_a_reduce_of_nothing — DUP — test_an_roi_with_no_live_pixel_fails_the_date_instead_of_dividing_by_zero (same windows=[] → (False, None)).

## tests/unit/ingest/test_source_coverage.py — TRIM — tests 15→14, LOC 265→~255, cpu-s 0.6→0.6
- test_every_probe_failing_is_inconclusive_and_never_raises — DUP — test_a_failed_block_probe_downgrades_absence_to_inconclusive.
Notes: test_a_stalled_probe_cannot_outrun_the_budget leaves a probe thread sleeping 30 s after the test returns; drop the sleep to ~2 s (still ≫ the 0.3 s budget) so it cannot hold a worker at session end. Inline `threading`/`time` imports → top.

## tests/unit/ingest/test_live_windows.py — TRIM — tests 29→24, LOC 415→~370, cpu-s 0.4→0.4
- TestMergeBands::test_extra_area_below_the_window_price_is_taken — DUP — test_matches_brute_force_optimum (120 random cases at prices 0/1/5/200).
- TestMergeBands::test_a_free_window_never_merges — DUP — same brute-force test includes price 0.
- TestMergeBands::test_expensive_area_splits_what_a_cheap_window_would_join — DUP — same; and test_per_date_narrowing_re_merges_at_the_run_s_price asserts the identical 5-vs-50 split.
- TestMergeBands::test_empty_input — PEDANTIC — empty in, empty out.
- TestGridFromChunkKeys::test_windows_are_identical_either_route — DUP — test_agrees_with_the_pixel_scan (grids equal ⇒ windows equal; same derivation follows).

## tests/unit/ingest/test_stac_streaming.py — TRIM — tests 30→15, LOC 446→~290, cpu-s 0.4→0.3
- TestMonthPartition (whole class, 12 tests incl. 6 param ids) — DUP — `month_ranges` partition/pad/leap/reversed are test_solar_days::test_owned_ranges_tile_the_window_exactly[month-*], test_the_query_pad_is_not_clamped_to_the_window[month], test_a_reversed_window_raises_rather_than_yielding_nothing.
- TestStreaming::test_a_single_day_window_streams — DUP — test_every_date_is_yielded_exactly_once (and solar_days single-day param).
- TestRetentionIsBounded::test_accumulating_across_months_is_what_streaming_prevents — PEDANTIC — "negative control" asserts a list the test itself built is non-empty; proves nothing about src.
- TestSolarDayOwnership::test_it_is_owned_exactly_once_either_way — DUP — the two preceding eastern/None tests + test_solar_days straddling-day property.
Notes: survivors cover what only this file covers: stream_stac_months ownership drop, per-month existing-date probe, depth-1 prefetch, failure propagation, weakref release, `_prefetch` daemon/reraise, and `_filter_existing_dates` solar keying.

## tests/unit/ingest/test_read_failure_verdict.py — TRIM (canonical home of the read-failure classifier) — tests 168→~65, LOC 400→~380, cpu-s 0.1→0.05
- test_no_cause_earns_both_verdicts (36 ids) — IMPROBABLE — its own docstring: "cannot fail by construction"; both predicates derive from one `ReadFailure` member (duplicates.py classify_read_failure_in).
- test_no_case_with_a_real_chain_is_ever_reported_as_flattened (36 ids) — PARAM/DUP — every row is a typed RasterioIOError, so `cause_was_flattened` is False on its first check; test_duplicate_granules::test_a_typed_exception_is_not_the_flattened_shape + test_a_chain_that_arrived_intact_is_not_flattened cover it.
- test_the_words_alone_reach_the_same_verdict — PARAM — 36 ids → 6 (one per verdict: ZIPDecode, ObjectNotFound, 401, 503, ExpiredToken, "Connection refused: ZIPDecode"); text and exception paths share classify_read_failure_in, only the text assembly differs.
- test_the_verdict_for_this_cause_has_not_moved — PARAM — rows "HTTP 502", "HTTP 504" (range-claimed like 599) and "HTTP 400" (same CLIENT_ERROR range as 401): 36 → 33.
Notes: this TABLE (36 rows, real chains, both-causes-in-one-chain precedence rows) supersedes the classifier tests in test_duplicate_granules — see the correction in Cross-file. Move here from test_duplicate_granules the cases the table cannot express: missing-object without reader corroboration (IcechunkError/NoSuchKey bare), NoSuchBucket, `response_code=` ranged wording (403/503) and `response_code=0`.

## tests/unit/ingest/test_opera_query.py — TRIM — tests 32→21, LOC 473→~360, cpu-s 0.1→0.1
- TestMgrsTileToBbox (4 tests) — DEAD — `mgrs_tile_to_bbox` has no src/scripts caller (grep: README mention only). Delete the function and its README line with it.
- TestMgrsTileToUtmEpsg (5 ids) — DEAD — `mgrs_tile_to_utm_epsg`, same.
- TestNormalizeOperaTimestamps::test_empty_input — PEDANTIC — empty in, empty out.
- TestNormalizeOperaTimestamps::test_modifies_in_place_and_returns_same_list — PEDANTIC — identity of the returned list.
- TestQueryCmrGranules::test_requires_both_polarisations_not_just_vv — PEDANTIC — asserts a constant; test_passes_attribute_filters asserts both VV and VH reach the request.
- TestQueryCmrGranules::test_normal_bbox_queries_once — MOCK — asserts a patched helper was called once; test_returns_built_items covers the unsplit path.
Notes: TestGranuleToItem::test_skip_reports_polarizations_found is half log-wording asserts ("EW-mode", "CMR reported", "inconsistent with its own metadata" absences) — keep the skip_counts assert, drop the six caplog text asserts (PEDANTIC). Inline import of `_REQUIRED_POLARIZATIONS` goes with its test.

## tests/unit/ingest/test_auth.py — TRIM — tests 18→14, LOC 423→~340, cpu-s 0.0
- test_get_edl_session_partial_basic_auth_raises — PARAM — fold into test_get_edl_session_no_creds_raises as a second env case.
- test_datapool_url_to_s3_flat_datapool_vh — PARAM — same suffix strip as the VV case; parametrize the band.
- test_build_aws_env_maps_keys — PEDANTIC — dict key renaming.
- test_odc_thread_session_explicit_reset_picks_up_env — DUP — exercises odc's own `_local.reset()`; test_odc_thread_session_tracks_env_after_drift proves our patch makes the reset unnecessary.
Notes: ~14 inline imports (`get_edl_session` x4, `_datapool_url_to_s3` x4, `threading` x3, odc `_local`, `auth` module) → top of file; the three odc tests share a "set three AWS env vars on a fresh thread" body → one helper (~40 LOC).

## tests/unit/ingest/test_source_read_resilience.py — TRIM — tests 14→8, LOC 377→~260, cpu-s 0.1→0.1
- test_the_retry_ladder_is_the_one_production_pays_for — PEDANTIC — pins tenacity parameters (1/2/15) and their 61 s sum.
- test_each_retry_is_logged_so_transient_and_permanent_are_distinguishable — PEDANTIC — counts WARNING records.
- test_context_is_transparent_when_nothing_fails — PEDANTIC — asserts no log records on success.
- TestTheContextCompletesTheReason::test_the_same_failure_with_nothing_logged_still_reads_as_bad_bytes — DUP — test_loader_failure_attribution TestTheReasonGdalDidNotRaise::test_without_the_log_the_refusal_reads_as_bad_imagery.
- TestZeroDateOutcomeIsAttributable::test_every_informational_line_carries_the_roi — PEDANTIC — regex over log format strings in s1_roi.py.
- TestEveryComputeThatReadsSourceIsAttributed::test_the_optical_write_passes_its_items — PEDANTIC — greps one literal `items=prepared.items`.
Notes: delete the six-line comment block about a test removed on 2026-08-25 (edit history). test_a_line_left_over_from_an_earlier_read sleeps 0.05 s — fine. TestZeroDateOutcomeIsAttributable becomes empty → remove the class.

## tests/unit/ingest/test_s1_given_up_dates.py — TRIM — tests 17→13, LOC 385→~320, cpu-s 0.3→0.25
- test_a_refusal_wrapped_in_a_warp_failure_is_still_a_refusal — DUP — test_a_refusal_fails_the_leg_rather_than_being_given_up[warp] (identical failure shape).
- test_a_refusal_fails_the_leg_rather_than_being_given_up[403-at-open] — PARAM — the classifier variety is test_read_failure_verdict's TABLE; keep [warp], [transport].
- test_a_failed_date_costs_only_itself[undecodable-at-open] — PARAM — same loop branch as [undecodable-in-warp].
- test_a_leg_that_gave_up_some_dates_and_committed_others_still_succeeds — DUP — test_a_failed_date_costs_only_itself (one date fails, rest written, status success).
Notes: test_past_the_ceiling_the_leg_stops... asserts five message fragments ("TERMINAL", "RE-DISPATCH", "reprocessed copies", ...) — keep the type + ceiling match, drop the wording asserts (PEDANTIC) and its "RENAMED" docstring history. Inline imports in `_run` (collections, numpy, xarray, s1_roi — the last already imported at top) → top.

## tests/unit/ingest/test_radar_free_zone.py — TRIM — tests 27→17, LOC 269→~180
- TestAbandoningAnUnusableOrbit (whole class, 4 tests) — MOCK — three tests exercise the test's own `_consume` helper (a rule that does not exist in src); the fourth asserts a constant is ABSENT via hasattr (PEDANTIC). Move the reasoning to context_docs if wanted.
- TestTheInferenceConfigAcceptsRadarFreeLand::test_the_forcing_is_logged_loudly — PEDANTIC — log wording.
- TestTheInferenceConfigAcceptsRadarFreeLand::test_the_forcing_is_reproducible_from_s1_orbit_alone — PEDANTIC — a pure constructor called twice returns the same field.
- TestTheInferenceConfigAcceptsRadarFreeLand::test_the_real_orbits_are_unaffected_by_the_flag (3) — DUP — test_a_real_orbit_never_has_the_flag_forced (same constructor, same orbits).
- test_the_demand_failure_says_where_to_confirm_which_case_it_is — PEDANTIC — error-message text.
Notes: this file tests inference/data_loading + config + flow defaults, not ingest/ — it belongs under tests/unit/inference/ (or config/). TestWhoDemandsRadarAndWhoDoesNot's two string-count checks (run_global_campaign, plain.py) are brittle literal greps; the AST default check is the right pattern.

## tests/unit/ingest/test_roi_processing.py — TRIM — tests 10→2, LOC 211→~60
- TestFilterLowCoverageDates (whole class, 7 tests) — DEAD — `filter_low_coverage_dates` has no src/scripts caller (grep `filter_low_coverage_dates(`: only this file; README + a comment mention it). Delete the function with its tests.
- TestApplyROIMask::test_custom_fill_value — DEAD — the one production caller (s1_roi.py:751) never passes fill_value.
Notes: both survivors patch `read_roi_mask` to reach the no-`roi_mask` fallback, which test_roi_mask_construction::test_every_caller_of_apply_roi_mask_supplies_the_mask proves production never takes; pass `roi_mask=mask_2d` instead and drop the patch (and consider deleting the fallback from src).

## tests/unit/ingest/test_item_baselines.py — TRIM (canonical home of baseline parsing) — tests 62→~24, LOC 172→~120
- TestARealVersionParses::test_every_known_form — PARAM — 13 ids → "05.10", "00.01", "00.00", "5.1", "4", 4.0, "  05.10  ", plus "99.99" (absorbs the grammar-bound test).
- TestAnythingElseIsUnreadable::test_it_reads_as_nothing — PARAM — 43 ids → one per category (~15): "NaN", float inf, "-1.00", "+4.00", "3.999", "3.99999999999999999999999999999", "1e308", "1e-1000000000", "100.00", "0x10", arabic-indic, "4.", "", "v4.00", None/True/[].
- TestARealVersionParses::test_a_declared_zero_is_not_unreadable — DUP — IS_A_VERSION "00.00" + the None id.
- TestARealVersionParses::test_a_one_digit_fraction_means_tenths — DUP — IS_A_VERSION "5.1".
- TestAnythingElseIsUnreadable::test_nothing_in_the_list_raises — DUP — the param test calls the same function on every entry; a raise fails it.
- TestAnythingElseIsUnreadable::test_the_grammar_bounds_the_value_without_a_range_check — DUP — "100.00" is in NOT_A_VERSION; move "99.99" into IS_A_VERSION.
Notes: `processing_baseline` here IS `stac._declared_baseline` and `duplicates.item_processing_baseline` (both are import aliases: stac.py:52, duplicates.py:60). So the parser tests in test_baseline_correction_by_source (TestTheBaselineParserReportsUnknown, TestScalingCannotOverflowThePicker), test_duplicate_granules (TestNonFiniteBaselinesAreUnknown param + real-baseline test) and test_stac_ingest (TestBaselineExtraction 400/510) are all DUP of this file — keep only `_extract_baseline`'s None→0 mapping (one test) and `extract_baselines`' date map elsewhere.

## tests/unit/ingest/test_ingest_roi_domain.py — DELETE (move 2 tests) — tests 8→0 here (2 move), LOC 205→0 (+~90 elsewhere)
- test_s2_result_has_expected_fields — PEDANTIC — dataclass field existence.
- test_s1_result_has_expected_fields — PEDANTIC — dataclass field existence.
- test_s1_orbit_literal_values — PEDANTIC — str() of a Literal alias.
- test_s2_signature_takes_client_and_logger — PEDANTIC — parameter names; every leg test calls with these.
- test_s1_signature_takes_credential_callbacks — PEDANTIC — parameter names.
- test_no_prefect_or_get_client_in_domain_modules — HISTORY — `.importlinter` (lines 37-47) forbids prefect imports outside orchestration/prefect for the whole package.
Notes: MOVE test_s1_batch_windows_tile_inclusive_range_without_overlap → test_s1_batch_lookahead.py (it is the S1 batch/provider-window wiring; seven stacked @patch decorators can use that file's `_run_ingest`-style monkeypatch helper). MOVE test_query_stac_items_passes_stac_io_to_client → test_catalogue_refusal.py (retry wiring; the only test that does not patch StacApiIO away).

## tests/unit/ingest/test_date_footprint.py — TRIM — tests 25→17, LOC 185→~170
- test_footprint_covers_every_pixel_it_claims — PARAM — 5 ids → unaligned, far-corner, whole-grid (origin and interior cells are subsumed by the unaligned rectangle).
- test_unusable_bbox_assumes_everything — PARAM — 5 ids → None, (1.0, 2.0), ("a","b","c","d").
- test_grid_round_trips_through_windows — PARAM — 5 seeds → 2.
- test_no_run_windows_stays_empty — PEDANTIC — empty in, empty out.

## tests/unit/ingest/test_s1_credential_expiry.py — TRIM — tests 12→10, LOC 139→~115
- test_margin_exceeds_the_fallback_cadence_is_not_required_but_margin_is_generous — PEDANTIC — asserts constants' ranges.
- test_tick_interval_is_well_inside_the_margin — PEDANTIC — constant inequality.
Notes: test_ticker_stops_when_the_block_exits sleeps 0.2 s; 0.05 s is still 5 tick intervals.

## tests/unit/ingest/test_roi.py — TRIM — tests 6→3, LOC 120→~90: TestReadROIMask (both) DUP of test_roi_mask_construction::test_the_mask_is_the_array_that_was_written (pixels + chunk layout); test_wgs84_zarr_returns_stored_bbox PARAM of the UTM case.
## tests/unit/ingest/test_transforms.py — TRIM — tests 12→11: test_preserves_coordinates PEDANTIC (coord names present).
## tests/unit/ingest/test_ingest_knob_plumbing.py — KEEP — 14 tests, 75 LOC; cheap mechanical signature parity that caught a real run-time failure.
## tests/unit/ingest/conftest.py — KEEP — `s1_leg_offline`, used by three S1 files.
## tests/stac_items.py — KEEP — `fake_stac_item`, used by grouping, s2_date_pipeline, streaming.

## Cross-file
- CORRECTION to the test_duplicate_granules section: test_read_failure_verdict.py's TABLE (36 real chains, incl. both-causes-in-one-chain precedence rows) already IS the precedence test. So delete ALL classifier classes from test_duplicate_granules — TestWhenToStepDown, TestWhenTheProviderRefused, TestAMessageMayNotBeItsOwnCorroboration, TestTheTwoVerdictsCannotDisagree, TestANamedTransportFailureIsNotBadData (85 tests) — and move only the cases the table cannot express into test_read_failure_verdict: bare missing-object without reader corroboration, NoSuchBucket, `response_code=` 403/503, `response_code=0`. Also delete TestNonFiniteBaselinesAreUnknown's parse tests (keep only the NaN order-independence selector test). Net for that file: 270→~120 tests, ~1,250 LOC.
- One parser, four test homes: `item_baselines.processing_baseline` is imported as `stac._declared_baseline` and `duplicates.item_processing_baseline`. Keep test_item_baselines; delete parse tests in test_baseline_correction_by_source, test_duplicate_granules, test_stac_ingest.
- One classifier, three homes: test_read_failure_verdict (canonical), test_duplicate_granules (delete), test_loader_failure_attribution / test_source_read_resilience (keep only the evidence-capture wiring).
- Solar-day ownership tested in test_solar_days AND re-tested in test_stac_streaming TestMonthPartition AND through the S1/S2 legs: keep test_solar_days as the library layer; legs keep one wiring test each.
- `pipelined` tested in test_pipeline and test_s1_batch_lookahead: keep test_pipeline only.
- Merges: three test_roi_mask_* → test_roi_mask.py; test_solar_day_grouping → test_solar_days; test_s2_coverage_windows → test_s2_coverage_gate; test_ingest_roi_domain split into test_s1_batch_lookahead + test_catalogue_refusal; delete test_solar_day_labelling. test_radar_free_zone belongs under tests/unit/inference (it tests data_loading/config/flows).
- DEAD src found: `filter_low_coverage_dates` (roi_processing), `mgrs_tile_to_bbox` / `mgrs_tile_to_utm_epsg` (opera_query), the `item_filter_fn` parameter (stac.query_stac_items/has_new_stac_dates/ingest_tile), `apply_roi_mask`'s no-mask fallback and `fill_value`. Delete with their tests.
- Slowness (whole slice ~36 cpu-s → ~15): test_roi_mask_credential_expiry (credential_process forks Python per resolution → shell issuer), test_roi_mask_construction (2x LocalCluster(processes=True) → 1, n_workers=1), test_s2_date_pipeline (BOTH_MODES doubling + write_s sleeps + threaded dask → serial-only for out-of-loop tests, write_s 0.03, synchronous scheduler), test_pipeline (1.8 s of sleeps → shrink 5x).
- Style debt across the slice: docstrings narrate edit history ("REVERSED", "Raised on PR #107", "USED TO", "RENAMED", dated removals) — owner's rule is current facts only; ~20% of slice LOC. Dozens of inline imports (auth, s1_*, boa_offset, stac_ingest, duplicate_granules).
