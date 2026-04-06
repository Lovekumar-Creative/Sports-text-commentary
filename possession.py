import math

def distance(p1,p2):

    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def get_ball_owner(players, ball):

    if ball is None:
        return None

    closest_player = None
    min_dist = float("inf")

    for player_id, pos in players:

        d = distance(pos, ball)

        if d < min_dist:
            min_dist = d
            closest_player = player_id

    return closest_player