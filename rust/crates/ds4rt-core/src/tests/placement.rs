use super::*;

#[test]
fn modulo_policy_matches_flash_formula() {
    let hosts = EXPERT_HOSTS
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        owner_for_expert(
            0,
            0,
            DS4_FLASH_ROUTED_EXPERTS,
            &hosts,
            PlacementPolicy::Modulo
        )
        .unwrap(),
        "spark-0"
    );
    assert_eq!(
        owner_for_expert(
            0,
            1,
            DS4_FLASH_ROUTED_EXPERTS,
            &hosts,
            PlacementPolicy::Modulo
        )
        .unwrap(),
        "spark-1"
    );
    assert_eq!(
        owner_for_expert(
            1,
            0,
            DS4_FLASH_ROUTED_EXPERTS,
            &hosts,
            PlacementPolicy::Modulo
        )
        .unwrap(),
        "spark-0"
    );
}

#[test]
fn range_policy_splits_256_experts_four_ways() {
    let hosts = EXPERT_HOSTS
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        owner_for_expert(7, 0, 256, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-0"
    );
    assert_eq!(
        owner_for_expert(7, 63, 256, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-0"
    );
    assert_eq!(
        owner_for_expert(7, 64, 256, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-1"
    );
    assert_eq!(
        owner_for_expert(7, 255, 256, &hosts, PlacementPolicy::Range).unwrap(),
        "spark-3"
    );
}

#[test]
fn range_policy_uses_pro_expert_count() {
    let hosts = EXPERT_HOSTS
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        owner_for_expert(60, 95, 384, &hosts, PlacementPolicy::Range).as_deref(),
        Some("spark-0")
    );
    assert_eq!(
        owner_for_expert(60, 96, 384, &hosts, PlacementPolicy::Range).as_deref(),
        Some("spark-1")
    );
    assert!(owner_for_expert(60, 384, 384, &hosts, PlacementPolicy::Range).is_none());
}
