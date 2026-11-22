"""Chess agent. Team: Team 06. Author: Laura Somogyi Student ID: 25055."""

'''
Start here: docs/students/README.md
Full rules: games/chess/README.md
Example of what you receive: docs/students/sample_observations/chess.json

You may use python-chess (`import chess`) to represent the board and generate moves.
You may not use chess.engine, Stockfish or any other existing chess engine.

Keep the function names and signatures below. Add any helpers you need.
'''

import chess
import time

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE = 100000
_TT = {}


def evaluate(fen):
    """Heuristic score of a position, from White's point of view."""
    board = chess.Board(fen)

    if board.is_checkmate():
        if board.turn == chess.BLACK:
            return MATE_SCORE - board.ply()
        return -MATE_SCORE + board.ply()
    if board.is_game_over(claim_draw=False):
        return 0

    score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        central_distance = abs(3.5 - file) + abs(3.5 - rank)

        if piece.piece_type == chess.PAWN:
            advancement = rank if piece.color == chess.WHITE else 7 - rank
            value += advancement * 6
        elif piece.piece_type == chess.KNIGHT or piece.piece_type == chess.BISHOP:
            value += (7 - central_distance) * 5
        elif piece.piece_type == chess.ROOK:
            value += 3 if rank in (0, 7) else 0

        score += value if piece.color == chess.WHITE else -value

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 25
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 25

    score += (board.legal_moves.count() if board.turn == chess.WHITE else
              -board.legal_moves.count()) * 2

    if board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE):
        score += 15
    if board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK):
        score -= 15

    return float(score)

def _ordered_moves(board):
    moves = list(board.legal_moves)

    def priority(move):
        score = 0
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            captured_value = PIECE_VALUES[captured.piece_type] if captured else 0
            attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
            score += 10 * captured_value - attacker_value
        if move.promotion:
            score += PIECE_VALUES[move.promotion]
        board.push(move)
        if board.is_check():
            score += 600
        board.pop()
        return score

    moves.sort(key=priority, reverse=True)
    return moves

def minimax(fen, depth):
    board = chess.Board(fen)
    nodes = 0

    def search(position, remaining):
        nonlocal nodes
        nodes += 1
        if remaining == 0 or position.is_game_over(claim_draw=False):
            return evaluate(position.fen())

        if position.turn == chess.WHITE:
            value = -float("inf")
            for move in position.legal_moves:
                position.push(move)
                value = max(value, search(position, remaining - 1))
                position.pop()
            return value

        value = float("inf")
        for move in position.legal_moves:
            position.push(move)
            value = min(value, search(position, remaining - 1))
            position.pop()
        return value

    return {"value": search(board, depth), "nodes": nodes}

def alphabeta(fen, depth):
    board = chess.Board(fen)
    nodes = 0

    def search(position, remaining, alpha, beta):
        nonlocal nodes
        nodes += 1
        if remaining == 0 or position.is_game_over(claim_draw=False):
            return evaluate(position.fen())

        if position.turn == chess.WHITE:
            value = -float("inf")
            for move in _ordered_moves(position):
                position.push(move)
                value = max(value, search(position, remaining - 1, alpha, beta))
                position.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = float("inf")
        for move in _ordered_moves(position):
            position.push(move)
            value = min(value, search(position, remaining - 1, alpha, beta))
            position.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    value = search(board, depth, -float("inf"), float("inf"))
    return {"value": value, "nodes": nodes}

def _search_root(board, depth, deadline):
    maximizing = board.turn == chess.WHITE
    best_move = None
    best_value = -float("inf") if maximizing else float("inf")
    alpha = -float("inf")
    beta = float("inf")

    for move in _ordered_moves(board):
        if time.perf_counter() >= deadline:
            return None, best_value, True
        board.push(move)
        value = _search_alpha_beta(board, depth - 1, alpha, beta, deadline)
        board.pop()
        if value is None:
            return None, best_value, True

        if (maximizing and value > best_value) or (not maximizing and value < best_value):
            best_value = value
            best_move = move
        if maximizing:
            alpha = max(alpha, best_value)
        else:
            beta = min(beta, best_value)

    return best_move, best_value, False

def _search_alpha_beta(board, remaining, alpha, beta, deadline):
    if time.perf_counter() >= deadline:
        return None
    if remaining == 0 or board.is_game_over(claim_draw=False):
        return evaluate(board.fen())

    if board.turn == chess.WHITE:
        value = -float("inf")
        for move in _ordered_moves(board):
            board.push(move)
            child = _search_alpha_beta(board, remaining - 1, alpha, beta, deadline)
            board.pop()
            if child is None:
                return None
            value = max(value, child)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = float("inf")
    for move in _ordered_moves(board):
        board.push(move)
        child = _search_alpha_beta(board, remaining - 1, alpha, beta, deadline)
        board.pop()
        if child is None:
            return None
        value = min(value, child)
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value

def act(observation):
    """Called once per move. Module-level variables survive between calls within a game.

    observation["fen"]                        the position
    observation["color"]                      "white" or "black"
    observation["legal_moves"]                legal moves in UCI, for example "e2e4"
    observation["history"]                    every move so far in UCI
    observation["ply"], ["is_check"]          public game state
    observation["game_number"], ["games_total"]
    observation["time_remaining_s"]           your clock (losing on time loses the game)
    observation["opponent_time_remaining_s"]  the rival's clock

    Return {"move": "e2e4"}, or {"move": "resign"} to resign.
    """
    position = chess.Board(observation["fen"])
    legal_moves = list(position.legal_moves)
    if not legal_moves:
        return {"move": "resign"}

    remaining = float(observation.get("time_remaining_s", 5.0))
    if remaining < 0.35:
        budget = 0.05
    else:
        budget = min(1.25, max(0.15, remaining * 0.12))
    deadline = time.perf_counter() + budget

    best_move = _ordered_moves(position)[0]
    if len(legal_moves) == 1:
        return {"move": best_move.uci()}

    for depth in range(1, 5):
        move, _value, timed_out = _search_root(position, depth, deadline)
        if timed_out:
            break
        if move is not None:
            best_move = move

    return {"move": best_move.uci()}