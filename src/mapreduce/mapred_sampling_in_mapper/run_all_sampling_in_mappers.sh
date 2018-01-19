HOSTAME=$(uname -n)
tempPath=$PATH
LOCAL_OUTPUT='output_sampling_in_mapper/sampling'

if [ $HOSTNAME == "Watchdog" ]; then
    HADOOP_HOME=/usr/local/hadoop/
    HADOOP_STREAM_JAR=${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar
elif [ $HOSTNAME == "watchdog" ]; then
    HADOOP_STREAM_JAR=${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar
    PATH=${PATH}:$HADOOP_HOME/bin
else
    HADOOP_HOME=/opt/cloudera/parcels/CDH/
    HADOOP_STREAM_JAR=${HADOOP_HOME}/jars/hadoop-streaming*mr*.jar
fi
HADOOP_BIN=${HADOOP_HOME}/bin/hadoop

if [ ! -f env.zip ]; then
    virtualenv env
    source env/bin/activate
    pip install gensim
    cd env
    zip -r ../env.zip *
    cd ..
fi

for file in {filenames_sampling_in_mapper.txt,article.txt,mapper_sampling_in_mappers.py,reducer.py}; do
    hdfs dfs -test -e $file
    if [ $? -ne 0 ]; then
        hdfs dfs -put $file;
    fi;
done

hdfs dfs -test -e output_sampling_in_mapper
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r output_sampling_in_mapper;
fi

echo $HADOOP_HOME
echo $HADOOP_STREAM_JAR

hadoop jar $HADOOP_STREAM_JAR\
    -archives ./env.zip#env\
    -file filenames_sampling_in_mapper.txt\
    -file mapper_sampling_in_mappers.py\
    -file reducer.py\
    -file article.txt\
    -input filenames_sampling_in_mapper.txt\
    -mapper mapper_sampling_in_mappers.py\
    -reducer reducer.py\
    -output output_sampling_in_mapper; 1>>mapred_std_out_sampling_in_mapper 2>>mapred_std_err_sampling_in_mapper

if [ -d output_sampling_in_mapper ]; then
    rm -r output_sampling_in_mapper;
fi

hdfs dfs -test -e output_sampling_in_mapper
if [ $? -eq 0 ]; then
    mkdir -pv $LOCAL_OUTPUT
    hdfs dfs -get output_sampling_in_mapper $LOCAL_OUTPUT
fi
PATH=$tempPath
