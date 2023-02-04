for url in $(cat dataset_url.txt)
do 
  file=${url##*/}
  if [ -e $file ]; then
    echo "dataset $file exists.Skipping..."
  else
    wget $url
  fi
done

mkdir -p Lip lammax
