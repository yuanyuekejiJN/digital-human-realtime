import sqlite3
import time
import threading
import functools
def synchronized(func):
  @functools.wraps(func)
  def wrapper(self, *args, **kwargs):
    with self.lock:
      return func(self, *args, **kwargs)
  return wrapper
class Question_Db:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.init_db()
   

    #初始化
    def init_db(self):
        conn = sqlite3.connect('fay.db')
        c = conn.cursor()
        try:
            conn.execute('SELECT * FROM T_Question')
        except sqlite3.OperationalError:
            c.execute('''CREATE TABLE T_Question
                        (id INTEGER PRIMARY KEY     autoincrement,
                        question           TEXT    NOT NULL,
                        answer           TEXT    NOT NULL,
                        createtime         Int);''')
            conn.commit()
        finally:
            conn.close()





    #添加对话
    @synchronized
    def add_content(self,question,answer):
        conn = sqlite3.connect("fay.db")
        cur = conn.cursor()
        cur.execute("insert into T_Question (question,answer,createtime) values (?,?,?)",
                    (question, answer, int(time.time())))

        conn.commit()
        conn.close()
        return cur.lastrowid



    #获取对话内容
    @synchronized
    def get_list(self,n,limit):
        conn = sqlite3.connect("fay.db")
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute("select id, question,answer,datetime(createtime, 'unixepoch', 'localtime') as timetext from T_Question "+" limit ?,?",(n,limit))

        list = cur.fetchall()
        conn.close()
        list = [dict(row) for row in list]

        return list

    @synchronized
    def get_count(self):
        conn = sqlite3.connect("fay.db")
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute(
            "select count(*) from T_Question ")
        result = cur.fetchone()
        # 关闭连接
        conn.close()
        return result[0]

    @synchronized
    def delete(self,id):
        conn = sqlite3.connect("fay.db")
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute(
            "DELETE FROM T_Question WHERE id = ?",(id,))
        conn.commit()
        # 关闭连接
        conn.close()

    @synchronized
    def edite(self,question):
        conn = sqlite3.connect("fay.db")
        conn.row_factory = sqlite3.Row

        cur = conn.cursor()

        cur.execute(
            "UPDATE T_Question set question = ? ,answer = ? WHERE id = ?",(question["question"],question["answer"],question["id"],))
        conn.commit()
        # 关闭连接
        conn.close()

question_db = Question_Db()

if __name__ == '__main__':
    db = Question_Db()
    db.init_db()
    db.add_content("你叫什么名字","我叫小元")
    print(db.get_list(0,2))

# a = Content_Db()
# s = a.get_list('all','desc',10)
# print(s)
   





   



