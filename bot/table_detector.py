from typing import Dict, Optional
import cv2

from bot.core.state import GameState


class TableDetector:
    def __init__(self, detectors: Dict[str, object]):
        self.detectors = detectors

    def process(self, table_bgr) -> GameState:
        gs = GameState()
        roi_gray = cv2.cvtColor(table_bgr, cv2.COLOR_BGR2GRAY)

        # Dealer
        dealer = self.detectors.get('dealer')
        if dealer:
            det = dealer.detect(roi_gray)
            if det:
                seat, score = det
                gs.dealer_seat = seat
                gs.confidences['dealer'] = score

        # Turn (next to act)
        turn = self.detectors.get('turn')
        if turn:
            det = turn.detect(roi_gray)
            if det:
                seat, score = det
                gs.next_to_act_seat = seat
                gs.confidences['turn'] = score

        # Cards
        cards = self.detectors.get('cards')
        if cards:
            holes, hs, board, bs = cards.detect(roi_gray)
            gs.hole_cards = holes
            gs.board_cards = board
            if hs:
                gs.confidences['hole_cards'] = min(1.0, sum(hs)/max(1, len(hs)))
            if bs:
                gs.confidences['board_cards'] = min(1.0, sum(bs)/max(1, len(bs)))

        # Numbers (pot)
        nums = self.detectors.get('numbers')
        if nums:
            det = nums.detect(roi_gray)
            if det:
                text, conf = det
                gs.pot = text
                gs.confidences['pot'] = conf

        return gs
