import traceback
import sys

from confluent_kafka.admin import NewTopic, AdminClient, KafkaException


print("Checking kafka...")

tries = [ "localhost", "127.0.0.1", "sv7-apu11", "elastic-similarity", "fvs-kafka" ]
for trie in tries:

	conf = {
	'bootstrap.servers': "%s:29094" % trie
	}
	print()
	print("before try- %s" % trie)
	sys.stdout.flush()
	sys.stderr.flush()

	try:
		admin = AdminClient(conf)
		topics = admin.list_topics(timeout=1)
		print("topics=", topics)
		print("ok.")
		sys.stdout.flush()
		sys.stderr.flush()
	except:
		print("FAIL")
		sys.stdout.flush()
		sys.stderr.flush()
		traceback.print_exc()	
	print("done try")

print()
print("Done.")

