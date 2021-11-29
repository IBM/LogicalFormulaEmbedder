wget -P ./datasets/ "http://cl-informatik.uibk.ac.at/cek/holstep/holstep.tgz"
tar -xvzf ./datasets/holstep.tgz -C ./datasets/
rm ./datasets/holstep.tgz
wget -P ./datasets/ "https://s3-us-west-2.amazonaws.com/data.allenai.org/downloads/SciTailV1.1.zip"
unzip ./datasets/SciTailV1.1.zip -d ./datasets/
rm ./datasets/SciTailV1.1.zip

