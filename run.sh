
SERVER=/ccxMLogE         #项目路径
cd $SERVER/CcxMLOGE-0.1.1/ccxMLogE

case "$1" in

 start)
   nohup python ccxModelApi.py 1>$SERVER/runinfo.log 2>&1 &
   echo "启动成功 $!"
   echo $! > $SERVER/server.pid
   ;;

 stop)
    kill `cat $SERVER/server.pid`
    rm -rf $SERVER/server.pid
    ;;

 restart)
   $0 stop
  sleep 1
  $0 start
  ;;


 *)
 echo "Usage: run.sh {start|stop|restart}"
   ;;

esac

exit 0

#nohup python ccxfpABSModelApi.py 1>$SERVER/FPinfo.log 2>$SERVER/FPerror.log &
