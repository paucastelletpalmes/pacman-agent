# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes.

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    """
    Devuelve los dos agentes que formarán el equipo.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    Clase base para nuestros agentes reflejo.
    Hereda de CaptureAgent y usa evaluate() con features y weights.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_successor(self, game_state, action):
        """Devuelve el estado sucesor tras ejecutar una accion."""
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def evaluate(self, game_state, action):
        """Puntuacion = suma ponderada (features * weights)."""
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def choose_action(self, game_state):
        """Escoge la accion legal con mayor puntuacion."""
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        # Si quedan solo 2 comidas o menos, corre directo a casa para asegurar la victoria
        food_left = len(self.get_food(game_state).as_list())
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}


class OffensiveAgent(ReflexCaptureAgent):
    """
    Agente Atacante:
    - Entra al campo enemigo y busca comida.
    - Huye de los fantasmas activos que esten a 5 casillas o menos.
    - Si lleva comida encima, le atrae volver a su lado a depositar puntos.
    - Busca capsulas si hay disponibles.
    - Evita detenerse (Directions.STOP).
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # Distancia a la comida mas cercana
        if food_list:
            min_food_dist = min(self.get_maze_distance(my_pos, f) for f in food_list)
            features['distance_to_food'] = min_food_dist

        # Si lleva bolitas de comida, incentivar regresar a base para puntuarlas
        carrying = my_state.num_carrying
        if carrying > 0:
            dist_home = self.get_maze_distance(my_pos, self.start)
            features['return_home'] = dist_home * carrying

        # Huir de fantasmas no asustados
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        active_ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        if active_ghosts:
            ghost_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in active_ghosts]
            closest_ghost = min(ghost_dists)
            if closest_ghost <= 5:
                features['danger'] = closest_ghost

        # Comer capsulas de poder
        capsules = self.get_capsules(game_state)
        if capsules:
            features['distance_to_capsule'] = min(self.get_maze_distance(my_pos, c) for c in capsules)

        # Penalizar quedarse quieto
        if action == Directions.STOP:
            features['stop'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'return_home': -2,
            'danger': 10,
            'distance_to_capsule': -1,
            'stop': -50,
        }


class DefensiveAgent(ReflexCaptureAgent):
    """
    Agente Defensor:
    - Se queda en nuestro territorio como fantasma.
    - Si un Pacman enemigo invade nuestro lado, lo persigue y lo elimina.
    - Evita quedarse quieto o cambiar de direccion erráticamente.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Prioridad: quedarse en su propio lado
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Detectar y perseguir invasores
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, e.get_position()) for e in invaders]
            features['invader_distance'] = min(dists)

        # Penalizar stop y reversa innecesaria
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'stop': -100,
            'reverse': -2,
        }
