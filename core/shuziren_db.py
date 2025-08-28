import sqlite3
import time
import threading
import functools

from utils import config_util


def synchronized(func):
  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    with self.lock:
      return func(self, *args, **kwargs)
  return wrapper


class Shuziren_Db:

    def init(self) -> None:
        self.lock = threading.Lock()
        if config_util.video_height == 3840:
            self.db = "fay_3840.db"
        else:
            self.db = "fay.db"
        self.init_db()


    #初始化
    def init_db(self):
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        try:
            conn.execute('SELECT * FROM T_Shuziren')
        except sqlite3.OperationalError:
            c.execute('''CREATE TABLE T_Shuziren
                        (id INTEGER PRIMARY KEY     autoincrement,
                        name           TEXT    NOT NULL,
                        sound           TEXT    NOT NULL,
                        image           TEXT    NOT NULL,
                        video           TEXT    NOT NULL,
                        video2           TEXT    NOT NULL,
                        createtime         Int);''')
            conn.commit()
        finally:
            conn.close()





    #添加对话
    @synchronized
    def add_content(self,name,sound,image='',video='',video2=''):
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("insert into T_Shuziren (name,sound,image,video,video2,createtime) values (?,?,?,?,?,?)",
                    (name,sound,image,video,video2 ,int(time.time())))

        conn.commit()
        conn.close()
        return cur.lastrowid



    #获取对话内容
    @synchronized
    def get_list(self,n,limit):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute("select id, name,sound,image,video,video2,datetime(createtime, 'unixepoch', 'localtime') as timetext from T_Shuziren "+" limit ?,?",(n,limit))

        list = cur.fetchall()
        conn.close()
        list = [dict(row) for row in list]

        return list

    @synchronized
    def get_count(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute(
            "select count(*) from T_Shuziren ")
        result = cur.fetchone()
        # 关闭连接
        conn.close()
        return result[0]

    @synchronized
    def delete(self,id):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute(
            "DELETE FROM T_Shuziren WHERE id = ?",(id,))
        conn.commit()
        # 关闭连接
        conn.close()

    @synchronized
    def edite(self,question):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute(
            "UPDATE T_Shuziren set name = ?,sound = ?,image = ?,video = ?,video2 = ? WHERE id = ?",(question["name"],question["sound"],"",question["video"],question["video2"],question["id"],))
        conn.commit()
        # 关闭连接
        conn.close()


shuziren_db = Shuziren_Db()

if __name__ == '__main__':
    db = Shuziren_Db()
    db.init_db()
    db.add_content("你叫什么名字","我叫小元")
    print(db.get_list(0,2))

# a = Content_Db()
# s = a.get_list('all','desc',10)
# print(s)
   





   



