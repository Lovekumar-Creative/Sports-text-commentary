def get_objects(results):

    boxes = results[0].boxes

    players = []
    ball = None

    if boxes.id is None:
        return players, ball

    for box, track_id, cls in zip(boxes.xyxy, boxes.id, boxes.cls):

        x1,y1,x2,y2 = map(int,box)

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        if int(cls) == 0:
            players.append((int(track_id),(cx,cy)))

        elif int(cls) == 32:
            ball = (cx,cy)

    return players, ball