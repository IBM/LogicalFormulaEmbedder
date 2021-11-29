cd /home/mvcro/.local/lib/python3.6/site-packages/amrlib
mkdir data
cd data
wget -P ./ "https://github.com/bjascob/amrlib-models/releases/download/model_parse_t5-v0_1_0/model_parse_t5-v0_1_0.tar.gz"
wget -P ./ "https://github.com/bjascob/amrlib-models/releases/download/model_generate_t5wtense-v0_1_0/model_generate_t5wtense-v0_1_0.tar.gz"
tar -xvf /mnt/d/Users/mvcro/Downloads/model_parse_t5-v0_1_0.tar.gz
tar -xvf /mnt/d/Users/mvcro/Downloads/model_generate_t5wtense-v0_1_0.tar.gz
mv model_parse_t5-v0_1_0 model_stog
mv model_generate_t5wtense-v0_1_0 model_gtos
rm model_parse_t5-v0_1_0.tar.gz
rm model_generate_t5wtense-v0_1_0.tar.gz
