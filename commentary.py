def generate_commentary(previous_owner, current_owner):

    if previous_owner is None:
        return None

    if previous_owner != current_owner:
        return f"Player {previous_owner} passes to Player {current_owner}"

    return None