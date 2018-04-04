HOSTAME=$(uname -n)
tempPath=$PATH
LOCAL_OUTPUT='output/sampling'
NPART=100
NDIM=500
TIMEOUT=10000000

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

for file in {input.txt,env.zip}; do
    hdfs dfs -test -e $file
    if [ $? -ne 0 ]; then
        hdfs dfs -put $file;
    fi;
done

hdfs dfs -rm mapper.py
hdfs dfs -put mapper.py

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r output;
fi

echo $HADOOP_HOME
echo $HADOOP_STREAM_JAR

hadoop jar $HADOOP_STREAM_JAR\
    -Dmapreduce.job.maps=$NPART -Dmapreduce.job.reduces=100 -Dmapreduce.map.memory.mb=10240\
    -Dmapreduce.map.java.opts=-Xmx8192m -Dmapreduce.task.timeout=$TIMEOUT\
    -archives ./env.zip#env\
    -files hdfs:///user/zijian/input.txt#input.txt,hdfs:///user/zijian/mapper.py#mapper.py\
    -input input.txt\
    -mapper "mapper.py $NPART $NDIM"\
    -reducer /bin/cat\
    -output output 1>>mapred_std_out.txt 2>>mapred_std_err.txt

hdfs dfs -test -e output
if [ $? -eq 0 ]; then
    if [ -d output ]; then
        rm -r output;
    fi
    mkdir -pv $LOCAL_OUTPUT
    hdfs dfs -get output $LOCAL_OUTPUT
fi
PATH=$tempPath
