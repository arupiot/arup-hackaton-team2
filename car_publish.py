from google.cloud import pubsub_v1
import json
from google.auth import jwt

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

# The `topic_path` method creates a fully qualified identifier
# in the form `projects/{project_id}/topics/{topic_name}`
topic_path = publisher.topic_path(project_id, topic_name)

for n in range(1, 10):
    data = u"Message number {}".format(n)
    # Data must be a bytestring
    data = data.encode("utf-8")
    # When you publish a message, the client returns a future.
    future = publisher.publish(topic_path, data=data)
    print(future.result())

print("Published messages.")
