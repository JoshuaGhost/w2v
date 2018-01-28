HOSTAME=$(uname -n)
tempPath=$PATH

EXP_NAME='dist_res_100'
INPUT='filenames_'${EXP_NAME}'.txt'
MAPPER='mapper_'${EXP_NAME}'.py'
REDUCER='reducer_'${EXP_NAME}'.py'
OUTPUT_FOLDER='output_'${EXP_NAME}
FILES_PREFIX='article_split'
STD_OUT=${EXP_NAME}"_stdout"
STD_ERR=${EXP_NAME}"_stderr"

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

for file in {$INPUT,${FILES_PREFIX}*,$MAPPER,$REDUCER}; do
    hdfs dfs -test -e $file
    if [ $? -ne 0 ]; then
        hdfs dfs -put $file;
    fi;
done

hdfs dfs -test -e $LOCAL_OUTPUT
if [ $? -eq 0 ]; then
    hdfs dfs -rm -r $LOCAL_OUTPUT;
fi

echo $HADOOP_HOME
echo $HADOOP_STREAM_JAR

hadoop jar $HADOOP_STREAM_JAR\
    -Dmapreduce.job.maps=10 -Dmapreduce.job.reduces=1 -archives ./env.zip#env\
    -file $INPUT\
    -file ${MAPPER}\
    -file ${REDUCER}\
    -file ${FILE_PREFIX}*\
    -input $INPUT\
    -mapper "${MAPPER} 422790"\
    -reducer "${REDUCER} 422790"\
    -output $OUTPUT_FOLDER; 1>>$STD_OUT 2>>$STD_ERR

if [ -d $OUTPUT_FOLDER ]; then
    rm -r $OUTPUT_FOLDER;
fi

hdfs dfs -test -e $OUTPUT_FOLDER
if [ $? -eq 0 ]; then
    mkdir -pv $LOCAL_OUTPUT
    hdfs dfs -get $OUTPUT_FOLDER $OUTPUT_FOLDER
fi
PATH=$tempPath
