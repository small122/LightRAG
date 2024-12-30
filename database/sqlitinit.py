import sqlite3

def initsqlit(filepath):
    # 连接到 SQLite 数据库（如果数据库不存在，会自动创建）
    conn = sqlite3.connect(filepath+'/filesystem.db')
    cursor = conn.cursor()
    # 创建知识库表 (KnowledgeBase)
    # 创建文档列表表 (DocumentList)

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS DocumentList (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        knowledge_base_id TEXT,
        FOREIGN KEY (knowledge_base_id) REFERENCES KnowledgeBase(id) ON DELETE CASCADE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS KnowledgeBase (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        document_list_id TEXT,
         file_count INTEGER DEFAULT 0,
        FOREIGN KEY (document_list_id) REFERENCES DocumentList(id) ON DELETE CASCADE
    )
    ''')

    # 创建文件表 (File)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS File (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        parsing_status int CHECK(parsing_status IN (-1, 0, 1)) NOT NULL,
        upload_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')


    # 创建文档列表与文件的关联表 (DocumentList_File)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS DocumentList_File (
        document_list_id TEXT,
        file_id TEXT,
        PRIMARY KEY (document_list_id, file_id),
        FOREIGN KEY (document_list_id) REFERENCES DocumentList(id) ON DELETE CASCADE,
        FOREIGN KEY (file_id) REFERENCES File(id) ON DELETE CASCADE
    )
    ''')
    # 创建 tasks 表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Tasks (
        taskid INTEGER PRIMARY KEY AUTOINCREMENT,
        taskname TEXT NOT NULL,
        description TEXT UNIQUE,
        status TEXT CHECK(status IN ('Pending', 'In Progress', 'Completed', 'Cancelled')) DEFAULT 'Pending',
        starttime DATETIME,
        endtime DATETIME
    );
    ''')
    # 创建会话列表数据库
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Conversations (
        id TEXT PRIMARY KEY,  -- SQLite 自动处理主键的唯一性和递增
        user_id INTEGER DEFAULT NULL,
        title TEXT DEFAULT NULL,  -- SQLite 没有 VARCHAR 类型，使用 TEXT 替代
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,  -- 使用 TEXT 存储时间戳
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        knowledge_base_id TEXT,
        FOREIGN KEY (knowledge_base_id) REFERENCES KnowledgeBase(id) ON DELETE CASCADE
    );
    ''')
    # 创建消息数据库
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Messages (
        id INTEGER PRIMARY KEY ,  -- SQLite 自动处理主键的唯一性和递增
        message_id TEXT NOT NULL,
        conversation_id TEXT NOT NULL,
        sender TEXT CHECK(sender IN ('user', 'ai')) NOT NULL,  -- 使用 TEXT 模拟 ENUM 类型
        content TEXT NOT NULL,
        mode TEXT CHECK(mode IN ('local', 'global', 'hybrid')) DEFAULT 'hybrid',  -- 使用 TEXT 模拟 ENUM 类型
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );
    ''')

    # 创建插入触发器：插入时更新文件数量
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS update_file_count_after_insert
        AFTER INSERT ON DocumentList_File
        FOR EACH ROW
        BEGIN
            UPDATE KnowledgeBase
            SET file_count = (SELECT COUNT(*) FROM DocumentList_File WHERE document_list_id = NEW.document_list_id)
            WHERE id = (SELECT knowledge_base_id FROM DocumentList WHERE id = NEW.document_list_id);
        END;
    ''')

    # 创建删除触发器：删除时更新文件数量
    cursor.execute('''
        CREATE TRIGGER IF NOT EXISTS update_file_count_after_delete
        AFTER DELETE ON DocumentList_File
        FOR EACH ROW
        BEGIN
            UPDATE KnowledgeBase
            SET file_count = (SELECT COUNT(*) FROM DocumentList_File WHERE document_list_id = OLD.document_list_id)
            WHERE id = (SELECT knowledge_base_id FROM DocumentList WHERE id = OLD.document_list_id);
        END;
    ''')


    # 提交事务并关闭连接
    conn.commit()
    cursor.close()
    conn.close()
    print("Tables created successfully.")
