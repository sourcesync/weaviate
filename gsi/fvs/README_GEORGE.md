
cleaaring the python training cache
* get docker id for python-training-postgres container
* exec bash into it:  docker exec -it <ID> /bin/bash
* get into psql prompt: psql -p 14032 -d caching_db -U fvs_post_user -W (use password: fvs_post_passwd)
* list tables: \dt
* delete all rows from tables: delete from <table_name>;

