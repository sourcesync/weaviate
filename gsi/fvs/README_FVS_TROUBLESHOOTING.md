
# FVS Troubleshooting

## GSI docker containers not properly starting (or constantly restarting):
* kill all docker containers: "docker rm $(docker ps -a -q)"
* locate the docker container run script and locate the final "docker-compose up" command and temporary remove "-d" if set
* run the script and you will see all the docker startup logs appear in the console which should allow you to easily find which container(s) are having issues
* once you fix the issue, you can restore the "-d" daemonize mode or run the non-daemonizing script behind the "screen" utility


## Training error shows error(s) on previously-run/old dataset ids
* clear the python training postgres tables
  * get docker id for python-training-postgres container
  * exec bash into it:  docker exec -it <ID> /bin/bash
  * get into psql prompt: "psql -p 14032 -d caching_db -U fvs_post_user -W" (use password: fvs_post_passwd)
  * list tables: \dt
  * delete all rows from tables: "delete from <table_name>;"
* you should consider also all removing all the entries from /home/public/elastic-similarity/python-training/cache/ since you just removed their tracking in FVS
* reboot and wait for all GSI docker containers to start

## Python training cache taking up too much disk space
* clear the python training postgres tables
  * get docker id for python-training-postgres container
  * exec bash into it:  docker exec -it <ID> /bin/bash
  * get into psql prompt: "psql -p 14032 -d caching_db -U fvs_post_user -W" (use password: fvs_post_passwd)
  * list tables: \dt
  * delete all rows from tables: "delete from <table_name>;"
* remove all the entries from /home/public/elastic-similarity/python-training/cache/ since you just removed their tracking in FVS
* reboot and wait for all GSI docker containers to start

