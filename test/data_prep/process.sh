/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname a9a >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname covtype --ext libsvm.binary.bz2  >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname phishing >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname rcv1_train --ext binary.bz2  --rename rcv1 >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname real-sim --ext bz2  >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname w8a >> process_log.txt

/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname avazu-app.tr --ext bz2 >> process_log.txt  
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname kdda --ext bz2  >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname news20 --ext binary.bz2  >> process_log.txt
/home/yutong/anaconda3/bin/python -m memory_profiler process_data.py --datasetname url_combined_normalized --ext bz2  --rename url_combined >> process_log.txt

/home/yutong/anaconda3/bin/python compute_Lip.py