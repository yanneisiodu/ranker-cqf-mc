from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

from logger import setup_logger
from prod_advanced_stack import AdvancedTradeEngine
from utils import get_output_dir, load_config, prepare_model_frame, save_json

logger = setup_logger(__name__)


class AgentDecisionService:
    """Deterministic tool layer for an LLM agent.

    This service never places trades. It only returns scored candidates, health reports,
    and execution-safe decision packets that a higher-level agent can consume.
    """

    def __init__(self, advanced_artifact_path: str, config_file: Optional[str] = None):
        self.config = load_config(config_file)
        self.engine = AdvancedTradeEngine.from_path(advanced_artifact_path, config_file=config_file)

    def health_packet_from_frame(self, frame) -> Dict[str, object]:
        scored = self.engine.score_frame(frame)
        health = self.engine.health_check(scored)
        return {
            "health": health,
            "rows": int(len(scored)),
            "date_min": str(scored["date"].min()) if len(scored) else None,
            "date_max": str(scored["date"].max()) if len(scored) else None,
        }

    def decision_packet_from_frame(self, frame, as_of_date: Optional[str] = None) -> Dict[str, object]:
        return self.engine.build_trade_plan(frame, as_of_date=as_of_date, enforce_watchdog=True)

    def health_packet_from_files(self, data_files: Sequence[str], include_targets: bool = False, nrows: Optional[int] = None) -> Dict[str, object]:
        frame = prepare_model_frame(data_files, self.config, include_targets=include_targets, nrows=nrows)
        return self.health_packet_from_frame(frame)

    def decision_packet_from_files(
        self,
        data_files: Sequence[str],
        as_of_date: Optional[str] = None,
        include_targets: bool = False,
        nrows: Optional[int] = None,
    ) -> Dict[str, object]:
        frame = prepare_model_frame(data_files, self.config, include_targets=include_targets, nrows=nrows)
        return self.decision_packet_from_frame(frame, as_of_date=as_of_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent-safe health and decision packet generator")
    parser.add_argument("--advanced-artifact", required=True, help="Path to the advanced stack artifact")
    parser.add_argument("--data", nargs="+", required=True, help="CSV file(s) with option snapshots")
    parser.add_argument("--config", default="./config.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Directory to write JSON packets")
    parser.add_argument("--as-of-date", default=None, help="Optional decision date")
    parser.add_argument("--health-only", action="store_true", help="Emit only the health packet")
    parser.add_argument("--include-targets", action="store_true", help="Whether to build target columns from the input data")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for smoke runs")
    args = parser.parse_args()

    service = AgentDecisionService(args.advanced_artifact, config_file=args.config)
    if args.health_only:
        packet = service.health_packet_from_files(args.data, include_targets=args.include_targets, nrows=args.nrows)
        filename = "agent_health_packet.json"
    else:
        packet = service.decision_packet_from_files(
            args.data,
            as_of_date=args.as_of_date,
            include_targets=args.include_targets,
            nrows=args.nrows,
        )
        filename = "agent_decision_packet.json"

    root = get_output_dir(service.config, args.output_dir)
    path = root / filename
    save_json(packet, path)
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
