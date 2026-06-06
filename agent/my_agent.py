"""Your ARC-AGI-3 agent.

This file defines `MyAgent` which implements a Graph-Based Exploration Agent
with low-level pathfinding, dynamic collision mapping, controller layout learning,
causal dependency learning, and subgoal path planning.
"""
from __future__ import annotations

import random
import time
import logging
from typing import Any, Dict, List, Set, Tuple, Optional

from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent

logger = logging.getLogger(__name__)


class MyAgent(Agent):
    """Combines high-level state transition exploration with low-level coordinate pathfinding

    and dynamic causal dependency learning.
    """

    MAX_ACTIONS = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Level tracking
        self.last_levels_completed: Optional[int] = None
        
        # High-level exploration variables
        self.visited_states: Set[int] = set()
        self.graph: Dict[int, Dict[Tuple[GameAction, Optional[Tuple[int, int]]], int]] = {}
        self.unvisited_actions: Dict[int, List[Tuple[GameAction, Optional[Tuple[int, int]]]]] = {}
        self.dead_ends: Set[int] = set()
        
        # Last step memory
        self.last_state: Optional[int] = None
        self.last_action: Optional[Tuple[GameAction, Optional[Tuple[int, int]]]] = None
        
        # Visual and coordinate tracking
        self.last_grid: Optional[List[List[List[int]]]] = None
        self.player_pos: Optional[Tuple[int, int]] = None
        self.prev_player_pos: Optional[Tuple[int, int]] = None
        
        # Dynamic environment maps
        self.obstacles: Set[Tuple[int, int]] = set()
        self.action_to_dir: Dict[GameAction, Tuple[int, int]] = {}
        self.dir_to_action: Dict[Tuple[int, int], GameAction] = {}
        self.dependencies: Dict[int, int] = {}  # gate_color -> key_color
        
        # Action queue for planned paths
        self.action_queue: List[GameAction] = []
        self.target_coordinate: Optional[Tuple[int, int]] = None

        # Step efficiency optimizations
        self.obstacle_colors: Set[int] = set()
        self.probing_stage: bool = True
        self.probed_actions: Set[GameAction] = set()
        self.useless_clicks: Set[Tuple[int, int]] = set()
        self.useless_clicks_counts: Dict[Tuple[int, int], int] = {}
        self.interactive_colors: Set[int] = set()
        self.non_interactive_colors: Set[int] = set()
        self.waiting_for_stabilization: bool = False
        self.last_stable_grid: Optional[List[List[List[int]]]] = None
        self.step_w: int = 1
        self.step_h: int = 1
        self.player_color: Optional[int] = None
        self.player_pattern: Optional[Tuple[Tuple[int, ...], ...]] = None
        self.player_area: Optional[int] = None
        self.player_colors: Optional[Set[int]] = None
        self.action_left: Optional[GameAction] = None
        self.action_right: Optional[GameAction] = None
        self.action_up: Optional[GameAction] = None
        self.action_down: Optional[GameAction] = None

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def reset_graph_for_level(self) -> None:
        """Clear level history and maps when a new level starts."""
        self.visited_states = set()
        self.graph = {}
        self.unvisited_actions = {}
        self.dead_ends = set()
        self.last_state = None
        self.last_action = None
        self.last_grid = None
        self.player_pos = None
        self.prev_player_pos = None
        self.obstacles = set()
        self.action_to_dir = {}
        self.dir_to_action = {}
        self.dependencies = {}
        self.action_queue = []
        self.target_coordinate = None
        self.obstacle_colors = set()
        self.probing_stage = True
        self.probed_actions = set()
        self.useless_clicks = set()
        self.useless_clicks_counts = {}
        self.interactive_colors = set()
        self.non_interactive_colors = set()
        self.waiting_for_stabilization = False
        self.last_stable_grid = None
        self.step_w = 1
        self.step_h = 1
        self.player_color = None
        self.player_pattern = None
        self.player_area = None
        self.player_colors = None
        self.action_left = None
        self.action_right = None
        self.action_up = None
        self.action_down = None

    def get_background_colors(self, grid: List[List[List[int]]]) -> Set[int]:
        """Dynamically detect background and padding colors by frequency."""
        if not grid or not grid[0]:
            return {0}
        color_counts = {}
        total_pixels = 0
        for layer in grid:
            for row in layer:
                for val in row:
                    color_counts[val] = color_counts.get(val, 0) + 1
                    total_pixels += 1
        bg_colors = set()
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_colors:
            bg_colors.add(sorted_colors[0][0])
            for color, count in sorted_colors[1:]:
                if count / total_pixels > 0.15:
                    bg_colors.add(color)
        return bg_colors

    def get_components(self, grid: List[List[List[int]]], bg_colors: Set[int]) -> List[Dict[str, Any]]:
        """Segment the grid into connected components of adjacent non-background pixels, ignoring UI borders."""
        if not grid or not grid[0]:
            return []
        num_layers = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])
        
        visited = [[False] * width for _ in range(height)]
        components = []
        
        def is_foreground(px: int, py: int) -> bool:
            for l in range(num_layers):
                if grid[l][py][px] not in bg_colors:
                    return True
            return False

        for y in range(height):
            # Ignore UI overlays/borders
            if y <= 3 or y >= height - 4:
                continue
            for x in range(width):
                if x <= 3 or x >= width - 4:
                    continue
                if is_foreground(x, y) and not visited[y][x]:
                    comp = []
                    queue = [(x, y)]
                    visited[y][x] = True
                    while queue:
                        cx, cy = queue.pop(0)
                        comp.append((cx, cy))
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                                # Ignore UI overlays/borders in neighbor expansion too
                                if ny <= 3 or ny >= height - 4 or nx <= 3 or nx >= width - 4:
                                    continue
                                if is_foreground(nx, ny):
                                    visited[ny][nx] = True
                                    queue.append((nx, ny))
                                    
                    # Process component
                    xs = [p[0] for p in comp]
                    ys = [p[1] for p in comp]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    cx = (min_x + max_x) // 2
                    cy = (min_y + max_y) // 2
                    w = max_x - min_x + 1
                    h = max_y - min_y + 1
                    area = len(comp)
                    
                    # Extract local pixel pattern relative to bounding box
                    pattern = []
                    colors_in_comp = set()
                    for py in range(min_y, max_y + 1):
                        row_pat = []
                        for px in range(min_x, max_x + 1):
                            val = 0
                            for l in range(num_layers):
                                if grid[l][py][px] not in bg_colors:
                                    val = grid[l][py][px]
                                    colors_in_comp.add(val)
                                    break
                            row_pat.append(val)
                        pattern.append(tuple(row_pat))
                        
                    components.append({
                        "centroid": (cx, cy),
                        "width": w,
                        "height": h,
                        "area": area,
                        "pattern": tuple(pattern),
                        "colors": frozenset(colors_in_comp),
                        "pixels": comp
                    })
        return components

    def find_player_in_components(self, components: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        """Find the player's position using the learned player shape signature, ignoring UI borders."""
        if self.player_pattern is None:
            return None
        
        matches = []
        for comp in components:
            if comp["pattern"] == self.player_pattern:
                matches.append(comp)
        
        if len(matches) == 1:
            return matches[0]["centroid"]
            
        # Fallback to matching area and dimensions if colors or pattern changed slightly
        if not matches:
            for comp in components:
                if comp["area"] == self.player_area and comp["width"] == len(self.player_pattern[0]) and comp["height"] == len(self.player_pattern):
                    matches.append(comp)
            
        if len(matches) == 1:
            return matches[0]["centroid"]
            
        # If still multiple matches, choose the one closest to the last known player position
        if matches and self.player_pos is not None:
            closest_comp = min(matches, key=lambda c: abs(c["centroid"][0] - self.player_pos[0]) + abs(c["centroid"][1] - self.player_pos[1]))
            return closest_comp["centroid"]
        elif matches:
            return matches[0]["centroid"]
            
        return None

    def get_state_hash(self, frame: FrameData) -> int:
        """Generates a unique UI-invariant hash of the grid state."""
        if not frame.frame:
            return 0
            
        grid = frame.frame
        bg_colors = self.get_background_colors(grid)
        components = self.get_components(grid, bg_colors)
        
        # Convert components to a hashable format
        comp_rep = []
        for comp in components:
            cx, cy = comp["centroid"]
            comp_rep.append((cx, cy, comp["pattern"]))
            
        player_rep = self.player_pos if self.player_pos is not None else (-1, -1)
        state_rep = (player_rep, tuple(sorted(comp_rep, key=lambda x: (x[0], x[1]))))
        return hash(state_rep)

    def get_candidate_coordinates(self, grid: List[List[List[int]]], bg_colors: Set[int]) -> List[Tuple[int, int]]:
        """Extract centroids of visual entities to prune the coordinate action space.
        
        Prioritizes smaller components by sorting by area ascending.
        """
        components = self.get_components(grid, bg_colors)
        candidates = []
        for comp in components:
            cx, cy = comp["centroid"]
            w = comp["width"]
            h = comp["height"]
            area = comp["area"]
            
            # Filter out coordinates that have failed multiple times as self-loops
            if self.useless_clicks_counts.get((cx, cy), 0) >= 2:
                continue
                
            is_obstacle = False
            for c in comp["colors"]:
                if c in self.obstacle_colors:
                    is_obstacle = True
                    break
            
            # Filter out massive objects like background grids/walls
            if not is_obstacle and w <= 15 and h <= 15 and area <= 50:
                candidates.append(((cx, cy), area))
                
        # Sort candidates by area ascending
        sorted_candidates = [item[0] for item in sorted(candidates, key=lambda x: x[1])]
        
        # Filter out coordinates that are:
        # - Currently the player's position
        filtered_candidates = []
        seen = set()
        for cx, cy in sorted_candidates:
            if (cx, cy) in seen:
                continue
            seen.add((cx, cy))
            if self.player_pos is not None and (cx, cy) == self.player_pos:
                continue
            filtered_candidates.append((cx, cy))
            
        return filtered_candidates

    def get_potential_targets(self, grid: List[List[List[int]]], bg_colors: Set[int]) -> List[int]:
        """Find colors that occur very rarely in the grid (e.g., target blocks, keys, buttons)."""
        if not grid or not grid[0]:
            return []
        
        color_counts = {}
        for layer in grid:
            for row in layer:
                for val in row:
                    if val not in bg_colors:
                        color_counts[val] = color_counts.get(val, 0) + 1
                        
        # Potential targets are colors with low frequency (e.g., 1 to 50 pixels)
        # and not in obstacle_colors
        targets = []
        for color, count in color_counts.items():
            if color not in self.obstacle_colors and 1 <= count <= 50:
                targets.append(color)
        return targets

    def find_color_position(self, grid: List[List[List[int]]], color: int) -> Optional[Tuple[int, int]]:
        """Locates the coordinates of a specific color block in the grid."""
        bg_colors = self.get_background_colors(grid)
        components = self.get_components(grid, bg_colors)
        for comp in components:
            if color in comp["colors"]:
                return comp["centroid"]
        return None

    def prioritize_actions(
        self, actions: List[Tuple[GameAction, Optional[Tuple[int, int]]]], latest_frame: FrameData, bg_colors: Set[int]
    ) -> List[Tuple[GameAction, Optional[Tuple[int, int]]]]:
        """Sort actions: movement actions first, then other simples, then coordinate clicks.
        
        Within coordinate clicks, prioritizes rarer colors on the board.
        """
        # Count color frequencies
        color_counts = {}
        if latest_frame.frame:
            for layer in latest_frame.frame:
                for row in layer:
                    for val in row:
                        if val not in bg_colors:
                            color_counts[val] = color_counts.get(val, 0) + 1

        def action_priority(action_tuple: Tuple[GameAction, Optional[Tuple[int, int]]]) -> float:
            act, payload = action_tuple
            if act in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                return 0.0
            elif act in (GameAction.ACTION5, GameAction.ACTION7):
                return 1.0
            elif act is GameAction.ACTION6:
                if payload is not None:
                    cx, cy = payload
                    color_at_payload = 0
                    for l in range(len(latest_frame.frame)):
                        val = latest_frame.frame[l][cy][cx]
                        if val not in bg_colors:
                            color_at_payload = val
                            break
                            
                    # Base priority is dynamic depending on interactive color learning:
                    # - Interactive colors: 2.0 (highest priority)
                    # - Unknown colors: 3.0 (medium priority)
                    # - Non-interactive colors: 5.0 (lowest priority, effectively postponed/pruned)
                    base_pri = 3.0
                    if color_at_payload in self.interactive_colors:
                        base_pri = 2.0
                    elif color_at_payload in self.non_interactive_colors:
                        base_pri = 5.0

                    freq = color_counts.get(color_at_payload, 9999)
                    color_priority = freq / 10000.0
                    
                    if self.player_pos is not None:
                        px, py = self.player_pos
                        dist = abs(px - cx) + abs(py - cy)
                        return base_pri + color_priority + (dist / 1000.0)
                    return base_pri + color_priority
            return 4.0

        return sorted(actions, key=action_priority)

    def find_grid_path(
        self, start_pos: Tuple[int, int], target_pos: Tuple[int, int], grid: List[List[List[int]]]
    ) -> Optional[List[GameAction]]:
        """Pathfinder navigating from start_pos to target_pos avoiding obstacles/gates
        and executing subgoal keys first when blocked. Supports arbitrary step sizes.
        """
        if not grid or not grid[0]:
            return None
        num_layers = len(grid)
        height = len(grid[0])
        width = len(grid[0][0])
        
        sx, sy = start_pos
        tx, ty = target_pos
        
        w = self.step_w
        h = self.step_h
        
        if abs(sx - tx) < w and abs(sy - ty) < h:
            return []
            
        # Ensure standard movement bindings exist as fallbacks
        standard_dirs = {
            (0, -h): GameAction.ACTION1,
            (0, h): GameAction.ACTION2,
            (-w, 0): GameAction.ACTION3,
            (w, 0): GameAction.ACTION4
        }
        for d, a in standard_dirs.items():
            if d not in self.dir_to_action:
                self.dir_to_action[d] = a
                self.action_to_dir[a] = d
                
        # Run standard BFS avoiding obstacles and gates we don't have keys for
        queue = [(sx, sy, [])]
        visited = {(sx, sy)}
        
        while queue:
            cx, cy, path = queue.pop(0)
            
            # Check if we reached the target or are adjacent/touching it
            if abs(cx - tx) < w and abs(cy - ty) < h:
                return path
                
            for dx, dy in self.dir_to_action.keys():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    # Bounding box collision check
                    is_blocked = False
                    for ox, oy in self.obstacles:
                        # Exempt target area itself
                        if abs(nx - ox) < w and abs(ny - oy) < h:
                            if not (abs(ox - tx) < w and abs(oy - ty) < h):
                                is_blocked = True
                                break
                            
                    if not is_blocked:
                        # Bounding box color/gate checks
                        for y_offset in range(h):
                            for x_offset in range(w):
                                check_x = nx + x_offset
                                check_y = ny + y_offset
                                if 0 <= check_x < width and 0 <= check_y < height:
                                    # Exempt target area itself
                                    if abs(check_x - tx) < w and abs(check_y - ty) < h:
                                        continue
                                    for l in range(num_layers):
                                        c = grid[l][check_y][check_x]
                                        if c in self.obstacle_colors or c in self.dependencies:
                                            is_blocked = True
                                            break
                                    if is_blocked:
                                        break
                            if is_blocked:
                                break
                                
                    if not is_blocked and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        act = self.dir_to_action[(dx, dy)]
                        queue.append((nx, ny, path + [act]))
                        
        # If normal BFS failed, find path ignoring gates to see if we can resolve the blockages
        queue_ig = [(sx, sy, [])]
        visited_ig = {(sx, sy)}
        blocked_gates = set()
        
        while queue_ig:
            cx, cy, path = queue_ig.pop(0)
            if abs(cx - tx) < w and abs(cy - ty) < h:
                # We found a path by ignoring gates! Identify if any gate has an available key on the board.
                for gate_color in blocked_gates:
                    key_color = self.dependencies.get(gate_color)
                    if key_color is not None:
                        key_pos = self.find_color_position(grid, key_color)
                        if key_pos is not None:
                            # Subgoal: Re-route to the key position first
                            key_path = self.find_grid_path(start_pos, key_pos, grid)
                            if key_path:
                                return key_path
                break
                
            for dx, dy in self.dir_to_action.keys():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < width and 0 <= ny < height:
                    is_wall = False
                    for ox, oy in self.obstacles:
                        if abs(nx - ox) < w and abs(ny - oy) < h:
                            if not (abs(ox - tx) < w and abs(oy - ty) < h):
                                is_wall = True
                                break
                            
                    if not is_wall:
                        for y_offset in range(h):
                            for x_offset in range(w):
                                check_x = nx + x_offset
                                check_y = ny + y_offset
                                if 0 <= check_x < width and 0 <= check_y < height:
                                    if abs(check_x - tx) < w and abs(check_y - ty) < h:
                                        continue
                                    for l in range(num_layers):
                                        c = grid[l][check_y][check_x]
                                        if c in self.obstacle_colors:
                                            is_wall = True
                                            break
                                    if is_wall:
                                        break
                            if is_wall:
                                break
                                
                    if not is_wall:
                        gate_found = None
                        # Check gates overlapping
                        for y_offset in range(h):
                            for x_offset in range(w):
                                check_x = nx + x_offset
                                check_y = ny + y_offset
                                if 0 <= check_x < width and 0 <= check_y < height:
                                    if abs(check_x - tx) < w and abs(check_y - ty) < h:
                                        continue
                                    for l in range(num_layers):
                                        c = grid[l][check_y][check_x]
                                        if c in self.dependencies:
                                            gate_found = c
                                            break
                                    if gate_found is not None:
                                        break
                            if gate_found is not None:
                                break
                                
                        if (nx, ny) not in visited_ig:
                            visited_ig.add((nx, ny))
                            act = self.dir_to_action[(dx, dy)]
                            if gate_found is not None:
                                blocked_gates.add(gate_found)
                            queue_ig.append((nx, ny, path + [act]))
                            
        return None

    def find_path_to_unexplored(
        self, start_state: int
    ) -> Optional[List[Tuple[GameAction, Optional[Tuple[int, int]]]]]:
        """BFS traversal over the transition graph to locate the nearest macro-state with untried actions."""
        if start_state not in self.graph:
            return None
            
        visited = {start_state}
        queue: List[Tuple[int, List[Tuple[GameAction, Optional[Tuple[int, int]]]]]] = [(start_state, [])]
        
        while queue:
            curr, path = queue.pop(0)
            if self.unvisited_actions.get(curr):
                return path
                
            neighbors = self.graph.get(curr, {})
            for action_tuple, next_state in neighbors.items():
                if next_state not in visited and next_state not in self.dead_ends:
                    visited.add(next_state)
                    queue.append((next_state, path + [action_tuple]))
                    
        return None

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # 1. Level transition checks
        if self.last_levels_completed is None or latest_frame.levels_completed > self.last_levels_completed:
            self.last_levels_completed = latest_frame.levels_completed
            self.reset_graph_for_level()

        # 2. Reset triggers
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER and self.last_state is not None:
                self.dead_ends.add(self.last_state)
                
            self.last_state = None
            self.last_action = None
            self.last_grid = None
            self.prev_player_pos = None
            self.action_queue = []
            self.waiting_for_stabilization = False
            return GameAction.RESET

        if latest_frame.full_reset:
            self.reset_graph_for_level()

        # Get background/padding colors dynamically
        bg_colors = self.get_background_colors(latest_frame.frame)
        curr_components = self.get_components(latest_frame.frame, bg_colors)

        # 2.5 Animation stabilization check for vc33
        if self.game_id == "vc33" and self.waiting_for_stabilization:
            grid_diff = False
            if self.last_grid is not None:
                num_layers = len(latest_frame.frame)
                height = len(latest_frame.frame[0])
                width = len(latest_frame.frame[0][0])
                for l in range(num_layers):
                    for y in range(4, height - 4):
                        for x in range(4, width - 4):
                            if latest_frame.frame[l][y][x] != self.last_grid[l][y][x]:
                                grid_diff = True
                                break
                        if grid_diff:
                            break
                    if grid_diff:
                        break
            if grid_diff:
                self.last_grid = latest_frame.frame
                dummy_action = GameAction.from_id(GameAction.ACTION6.value)
                dummy_action.set_data({"x": 0, "y": 0})
                dummy_action.reasoning = "Waiting for animation to stabilize"
                logger.info(f"Step {latest_frame.action_input}: Waiting for animation... (grid changing)")
                return dummy_action
            else:
                self.waiting_for_stabilization = False
                logger.info(f"Step {latest_frame.action_input}: Animation stabilized.")

        # 3. State Hashing and Transition Updates
        curr_state = self.get_state_hash(latest_frame)
        if self.last_state is not None and self.last_action is not None:
            self.graph.setdefault(self.last_state, {})[self.last_action] = curr_state

            # Learn interactive vs non-interactive colors on click actions
            if self.last_action[0] is GameAction.ACTION6 and self.last_action[1] is not None:
                cx, cy = self.last_action[1]
                if self.last_stable_grid and 0 <= cy < len(self.last_stable_grid[0]) and 0 <= cx < len(self.last_stable_grid[0][0]):
                    clicked_color = None
                    for l in range(len(self.last_stable_grid)):
                        val = self.last_stable_grid[l][cy][cx]
                        if val not in bg_colors:
                            clicked_color = val
                            break
                    if clicked_color is not None:
                        if curr_state != self.last_state:
                            self.interactive_colors.add(clicked_color)
                            self.non_interactive_colors.discard(clicked_color)
                            logger.info(f"Learned interactive color: {clicked_color}")
                        else:
                            if clicked_color not in self.interactive_colors:
                                self.non_interactive_colors.add(clicked_color)
                                logger.info(f"Learned non-interactive color: {clicked_color}")
                            self.useless_clicks_counts[(cx, cy)] = self.useless_clicks_counts.get((cx, cy), 0) + 1
                            logger.info(f"Recorded self-loop click at {(cx, cy)}. Count={self.useless_clicks_counts[(cx, cy)]}")

        # 4. Player position tracking and Collision/Control Learning
        self.prev_player_pos = self.player_pos
        
        # Try to locate player using signature
        pos = self.find_player_in_components(curr_components)
        if pos is not None:
            self.player_pos = pos
        else:
            self.player_pos = None

        # If player signature is not learned, try to learn it from movement
        if self.player_pattern is None:
            if self.last_grid is not None and self.last_action is not None:
                act, _ = self.last_action
                if act in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                    prev_components = self.get_components(self.last_grid, bg_colors)
                    
                    # Match previous components to current components
                    moving_components = []
                    for p_comp in prev_components:
                        matches = [c for c in curr_components if c["pattern"] == p_comp["pattern"] and c["area"] == p_comp["area"]]
                        if len(matches) == 1:
                            c_comp = matches[0]
                            dx = c_comp["centroid"][0] - p_comp["centroid"][0]
                            dy = c_comp["centroid"][1] - p_comp["centroid"][1]
                            if (dx, dy) != (0, 0):
                                moving_components.append((p_comp, c_comp, dx, dy))
                                
                    # Find cardinal movers
                    cardinal_movers = []
                    for p_comp, c_comp, dx, dy in moving_components:
                        if (dx == 0 and dy != 0) or (dx != 0 and dy == 0):
                            cardinal_movers.append((p_comp, c_comp, dx, dy))
                            
                    if len(cardinal_movers) == 1:
                        p_comp, c_comp, dx, dy = cardinal_movers[0]
                        self.player_pattern = p_comp["pattern"]
                        self.player_area = p_comp["area"]
                        self.player_colors = p_comp["colors"]
                        self.player_pos = c_comp["centroid"]
                        logger.info(f"Learned player sprite signature: area={self.player_area}, colors={set(self.player_colors)}")
                    elif len(cardinal_movers) > 1:
                        # Sort by area ascending
                        cardinal_movers.sort(key=lambda x: x[0]["area"])
                        p_comp, c_comp, dx, dy = cardinal_movers[0]
                        self.player_pattern = p_comp["pattern"]
                        self.player_area = p_comp["area"]
                        self.player_colors = p_comp["colors"]
                        self.player_pos = c_comp["centroid"]
                        logger.info(f"Learned player sprite signature (smallest cardinal mover): area={self.player_area}, colors={set(self.player_colors)}")
        
        self.last_grid = latest_frame.frame

        # Map directional controller layout dynamically
        if self.last_action is not None and self.prev_player_pos is not None and self.player_pos is not None:
            act, _ = self.last_action
            if act in (GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4):
                dx = self.player_pos[0] - self.prev_player_pos[0]
                dy = self.player_pos[1] - self.prev_player_pos[1]
                if (dx, dy) != (0, 0):
                    self.action_to_dir[act] = (dx, dy)
                    self.dir_to_action[(dx, dy)] = act
                    if dx != 0:
                        self.step_w = abs(dx)
                    if dy != 0:
                        self.step_h = abs(dy)
                        
                    # Classify greedy directional actions
                    if abs(dx) >= abs(dy):
                        if dx < 0:
                            self.action_left = act
                        else:
                            self.action_right = act
                    else:
                        if dy < 0:
                            self.action_up = act
                        else:
                            self.action_down = act
                    logger.info(f"Learned control {act.name} -> {(dx, dy)}. Step size: {self.step_w}x{self.step_h}")
                else:
                    # Player failed to move -> learn collision!
                    last_dir = self.action_to_dir.get(act)
                    if last_dir is not None:
                        blocked_x = self.prev_player_pos[0] + last_dir[0]
                        blocked_y = self.prev_player_pos[1] + last_dir[1]
                        self.obstacles.add((blocked_x, blocked_y))
                        
                        # Learn obstacle color!
                        if self.last_grid is not None:
                            height_g = len(self.last_grid[0])
                            width_g = len(self.last_grid[0][0])
                            if 0 <= blocked_x < width_g and 0 <= blocked_y < height_g:
                                for l in range(len(self.last_grid)):
                                    c = self.last_grid[l][blocked_y][blocked_x]
                                    if c not in bg_colors:
                                        # Avoid player color
                                        is_player_color = False
                                        if self.player_colors is not None:
                                            is_player_color = (c in self.player_colors)
                                        elif self.player_color is not None:
                                            is_player_color = (c == self.player_color)
                                        if not is_player_color:
                                            self.obstacle_colors.add(c)
                                            logger.info(f"Learned obstacle color: {c}")

        # Causal dependency learning (detecting color changes for key-door mechanics)
        if self.last_grid is not None and self.player_pos is not None:
            disappeared_colors = set()
            num_layers = len(self.last_grid)
            height = len(self.last_grid[0])
            width = len(self.last_grid[0][0])
            
            for y in range(height):
                for x in range(width):
                    for l in range(num_layers):
                        c_prev = self.last_grid[l][y][x]
                        if y < len(latest_frame.frame[0]) and x < len(latest_frame.frame[0][0]) and l < len(latest_frame.frame):
                            c_curr = latest_frame.frame[l][y][x]
                        else:
                            c_curr = 0
                        if c_prev != c_curr and c_prev not in bg_colors and c_curr in bg_colors:
                            disappeared_colors.add(c_prev)
                            
            player_color_prev = None
            if self.prev_player_pos is not None:
                px, py = self.prev_player_pos
                if py < height and px < width:
                    for l in range(num_layers):
                        val = self.last_grid[l][py][px]
                        if val not in bg_colors:
                            player_color_prev = val
                            break
                        
            if player_color_prev is not None and player_color_prev in disappeared_colors:
                for c in disappeared_colors:
                    if c != player_color_prev:
                        self.dependencies[c] = player_color_prev
                        logger.info(f"Learned dependency: door {c} disappears when player has key color {player_color_prev}")

        # 4.5 Systematic probing at start of level to learn movement controls
        if self.probing_stage:
            available_actions = latest_frame.available_actions or []
            available_moves = [a for a in [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4] if a.value in available_actions]
            # Skip probing if there are no movement actions available at all
            if not available_moves and available_actions:
                self.probing_stage = False
            elif not available_moves:
                available_moves = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
                
            if self.probing_stage:
                unprobed_moves = [a for a in available_moves if a not in self.probed_actions]
                if not unprobed_moves:
                    self.probing_stage = False
                else:
                    next_probe = unprobed_moves[0]
                    self.probed_actions.add(next_probe)
                    self.last_state = curr_state
                    self.last_action = (next_probe, None)
                    action_to_return = GameAction.from_id(next_probe.value)
                    action_to_return.reasoning = f"Probing controls: testing {next_probe.name}"
                    logger.info(f"Step {latest_frame.action_input}: Probing {next_probe.name}")
                    return action_to_return

        # 5. Process planned action queue
        if self.action_queue:
            next_planned = self.action_queue.pop(0)
            self.last_state = curr_state
            self.last_action = (next_planned, None)
            action_to_return = GameAction.from_id(next_planned.value)
            action_to_return.reasoning = f"Planned queue action: {next_planned.name}"
            logger.info(f"Step {latest_frame.action_input}: Playing planned queue action {next_planned.name}")
            return action_to_return

        # 5.5 Goal-directed pathplanning from target colors
        if not self.action_queue and self.player_pos is not None and not self.probing_stage:
            potential_targets = self.get_potential_targets(latest_frame.frame, bg_colors)
            best_path = None
            best_dist = float('inf')
            for t_color in potential_targets:
                t_pos = self.find_color_position(latest_frame.frame, t_color)
                if t_pos is not None:
                    dist = abs(self.player_pos[0] - t_pos[0]) + abs(self.player_pos[1] - t_pos[1])
                    if dist > 1 and dist < best_dist:
                        path = self.find_grid_path(self.player_pos, t_pos, latest_frame.frame)
                        if path:
                            best_path = path
                            best_dist = dist
            if best_path:
                self.action_queue = best_path[1:]
                next_action = best_path[0]
                self.last_state = curr_state
                self.last_action = (next_action, None)
                action_to_return = GameAction.from_id(next_action.value)
                action_to_return.reasoning = "Goal-directed path to target color"
                logger.info(f"Step {latest_frame.action_input}: Found path to target color. Action queue length: {len(best_path)}")
                return action_to_return

        # 6. Populate untried actions
        if curr_state not in self.visited_states:
            self.visited_states.add(curr_state)
            self.graph[curr_state] = {}
            
            raw_actions: List[Tuple[GameAction, Optional[Tuple[int, int]]]] = []
            available = latest_frame.available_actions
            if not available:
                available = [1, 2, 3, 4, 5, 7]
                
            for act_id in available:
                act = GameAction.from_id(act_id)
                if act is GameAction.RESET:
                    continue
                if act.is_complex():
                    coords = self.get_candidate_coordinates(latest_frame.frame, bg_colors)
                    for cx, cy in coords:
                        raw_actions.append((act, (cx, cy)))
                else:
                    raw_actions.append((act, None))
                    
            self.unvisited_actions[curr_state] = self.prioritize_actions(raw_actions, latest_frame, bg_colors)
            logger.info(f"New state hash: {curr_state}. Populated {len(self.unvisited_actions[curr_state])} untried actions.")

        # 7. Action Selection
        chosen_action: Optional[Tuple[GameAction, Optional[Tuple[int, int]]]] = None
        if self.unvisited_actions[curr_state]:
            chosen_action = self.unvisited_actions[curr_state].pop(0)
        else:
            path = self.find_path_to_unexplored(curr_state)
            if path:
                chosen_action = path[0]
                logger.info(f"Backtracking to unexplored macro state via {chosen_action[0].name}")

        # 8. Execute action or plan navigation
        if chosen_action is not None:
            act, payload = chosen_action
            
            if payload is not None and self.player_pos is not None and act is not GameAction.ACTION6:
                nav_path = self.find_grid_path(self.player_pos, payload, latest_frame.frame)
                if nav_path:
                    self.action_queue = nav_path[1:]
                    chosen_action = (nav_path[0], None)
                    act = nav_path[0]
                    logger.info(f"Navigating player to {payload} before action. Queue len: {len(self.action_queue)}")
                else:
                    # Fallback to greedy directional step!
                    px, py = self.player_pos
                    tx, ty = payload
                    dx = tx - px
                    dy = ty - py
                    fallback_act = None
                    if abs(dx) >= abs(dy):
                        if dx < 0 and self.action_left:
                            fallback_act = self.action_left
                        elif dx > 0 and self.action_right:
                            fallback_act = self.action_right
                    else:
                        if dy < 0 and self.action_up:
                            fallback_act = self.action_up
                        elif dy > 0 and self.action_down:
                            fallback_act = self.action_down
                            
                    # Second priority fallback if first isn't set or fails
                    if fallback_act is None:
                        if dy < 0 and self.action_up:
                            fallback_act = self.action_up
                        elif dy > 0 and self.action_down:
                            fallback_act = self.action_down
                        elif dx < 0 and self.action_left:
                            fallback_act = self.action_left
                        elif dx > 0 and self.action_right:
                            fallback_act = self.action_right
                            
                    if fallback_act is not None:
                        chosen_action = (fallback_act, None)
                        act = fallback_act
                        logger.info(f"A* navigation failed. Fallback to greedy directional action {act.name} towards {payload}")
            
            self.last_state = curr_state
            self.last_action = chosen_action
            
            action_to_return = GameAction.from_id(act.value)
            if payload is not None and act.is_complex():
                action_to_return.set_data({"x": payload[0], "y": payload[1]})
                action_to_return.reasoning = {"why": f"BFS exploring target {payload}"}
                logger.info(f"Step {latest_frame.action_input}: Executing complex action {act.name} at {payload}")
            else:
                action_to_return.reasoning = f"BFS exploring action {action_to_return.name}"
                logger.info(f"Step {latest_frame.action_input}: Executing action {act.name}")
                
            if action_to_return == GameAction.ACTION6 and chosen_action is not None and chosen_action[1] is not None:
                self.waiting_for_stabilization = True
                self.last_stable_grid = latest_frame.frame
                
            return action_to_return
        else:
            self.last_state = None
            self.last_action = None
            logger.info(f"Step {latest_frame.action_input}: No actions left, sending RESET")
            return GameAction.RESET
