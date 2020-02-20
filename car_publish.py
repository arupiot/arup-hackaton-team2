import json
from google.cloud import pubsub_v1
from google.auth import jwt
from jinja2 import Environment, FileSystemLoader
import random, os
from datetime import datetime as dt
from datetime import timedelta as td
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

project_id = "team-2-266717"
# topic_name = "projects/%s/topics/car" % project_id
topic_name = "car"

service_account_info = json.load(open("service-account-info.json"))
audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"

credentials = jwt.Credentials.from_service_account_info(
    service_account_info, audience=audience
)

publisher_audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
credentials_pub = credentials.with_claims(audience=publisher_audience)

publisher = pubsub_v1.PublisherClient(credentials=credentials_pub)


env = Environment(loader=FileSystemLoader(os.path.join(THIS_DIR,"templates")),
                     trim_blocks=True)

template = env.get_template('json_template.json')

# {
#   "id": {{ version }},
#   "timestamp": {{ timestamp }},
#   "points": {
#     "speed": {{ speed }},
#     "engine_load": {{ engine_load }},
#     "fuel_level": {{ fuel_level }},
#     "throttle_position": {{ throttle_position }},
#     "engine_coolant_temp": {{ engine_coolant_temp }},
#     "engine_rpm": {{ engine_rpm }},
#     "intake_air_temp": {{ intake_air_temp }},
#     "ambient_air_temp": {{ ambient_air_temp }},
#     "horsepower": {{ horsepower }}
#   }
# }

# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_name}`
topic_path = publisher.topic_path(project_id, topic_name)

n = 0
while True:
    # data = u"Message number {}".format(n)
    speed = random.uniform(0, 100)
    engine_load = random.uniform(0, 100)
    fuel_level = random.uniform(0, 100)
    throttle_position = random.uniform(0, 100)
    engine_coolant_temp = random.uniform(0, 100)
    engine_rpm = random.uniform(0, 5000)
    intake_air_temp = random.uniform(0, 50)
    ambient_air_temp = random.uniform(0, 50)
    horsepower = random.uniform(0, 500)
    today = dt.now().strftime("%Y-%m-%dT%H:%M:%S+01:00")
    data = template.render(id=random.randint(0,5),
                           speed=speed,
                           engine_load=engine_load,
                           timestamp=today,
                           fuel_level=fuel_level,
                           throttle_position=throttle_position,
                           engine_coolant_temp=engine_coolant_temp,
                           engine_rpm=engine_rpm,
                           intake_air_temp=intake_air_temp,
                           ambient_air_temp=ambient_air_temp,
                           horsepower=horsepower,
                           ).replace("\n","")
    # Data must be a bytestring
    data = data.encode("utf-8")
    print(data)
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data=data)
    print(future.result())
    time.sleep(0.5)
    n += 1

print("Published messages.")
