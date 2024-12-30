from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import os
from datetime import datetime
from starlette.responses import JSONResponse

#from examples.graph_visual_with_html import visual_with_html
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding, ollama_model_complete
from lightrag.utils import EmbeddingFunc
import numpy as np
from typing import Optional
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
import textract
import uuid
import sqlite3
from database.sqlitinit import initsqlit
# Load environment variables
load_dotenv()

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Default directory for RAG index
DEFAULT_RAG_DIR = "index_default"
WORKING_DIR = os.getenv("RAG_DIR", DEFAULT_RAG_DIR)
UPLOAD_DIR=os.getenv("UPLOAD_DIR")
SQLITE_DIR=os.getenv("SQLITE_DIR")
# Ensure the working directory exists
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(SQLITE_DIR):
    os.makedirs(SQLITE_DIR)
# 数据库初始化
# 创建或连接到 SQLite 数据库文件
initsqlit(SQLITE_DIR)
conn = sqlite3.connect(SQLITE_DIR+'/filesystem.db')  # 连接到数据库，如果没有会自动创建


# Initialize FastAPI app
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains, can be restricted to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Configure working directory
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# LLM model function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        **kwargs,
    )

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="BAAI/bge-m3",
        # model="acge_text_embedding",
        # api_key='EMPTY',
        api_key=os.getenv("SILICON_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
        #
    )


# Initialize RAG instance
#OPenai 写法
rag =None
workdir=''
#ollama写法
# rag = LightRAG(
#     working_dir=WORKING_DIR,
#     llm_model_func=ollama_model_complete,
#     llm_model_name="phi3:latest",
#     llm_model_max_async=4,
#     llm_model_max_token_size=32768,
#     llm_model_kwargs={"host": "http://223.2.37.45:11434", "options": {"num_ctx": 32768}},
#     embedding_func=EmbeddingFunc(
#         embedding_dim=1024, max_token_size=1024, func=embedding_func
#     ),
# )

# Data models for API requests
class MessageRequest(BaseModel):
    conversation:str
class KnowledgebaseRequest(BaseModel):
    name:str
class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    conversation_id:str
    knowledge_base_id:str
    only_need_context: bool = False

class InsertRequest(BaseModel):
    text: str

class InsertFileRequest(BaseModel):
    filename: str

class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None
class DeleteEnetityRequest(BaseModel):
    enetityname :str
class DeleteConversationRequest(BaseModel):
    conversationid:str
class GetdocumentRequest(BaseModel):
    knowledge_base_id: str
    page: int
    page_size: int
def changrag(workingdir):
    global rag
    if not os.path.exists(workingdir):
        os.makedirs(workingdir)
    rag = LightRAG(
        working_dir=workingdir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=1024, func=embedding_func
        ),
    )
# API routes
#更换rag目录
@app.post("/changeworkingdir")
async def changeworkdingdir(WORKING_DIR:str):
    try:
        if WORKING_DIR==workdir:
            return Response(status="success", message="无需更换")
        changrag(WORKING_DIR)
        return Response(status="success", message="更换成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

#创建新会话
@app.get("/conversationlist")
async def query_conversation():
    cursor=conn.cursor()
    cursor.execute("SELECT * FROM Conversations")
    conversationlist=cursor.fetchall()
    cursor.close()
    conversations=[{"id":row[0],"title":row[2],"created_at":row[3],"updated_at":row[4],"knowledge_base_id":row[5]}for row in conversationlist]
    return JSONResponse(content={"status":"success","data":conversations})

@app.post("/meassagelist")
async def query_message(request:MessageRequest):
    cursor=conn.cursor()
    cursor.execute("SELECT * FROM Messages WHERE conversation_id = ?", (str(request.conversation),))

    messagelist=cursor.fetchall()
    cursor.close()
    messages=[{"id":row[0],"message_id":row[1],"conversation_id":row[2],"sender":row[3],"content":row[4],"mode":row[5],"created_at":row[6]}for row in messagelist]
    return JSONResponse(content={"status":"success","data":messages})
@app.post("/createknowledgebase")
async def create_knoeledgebase(request:KnowledgebaseRequest):
    id=str(uuid.uuid4())
    documentlistid = str(uuid.uuid4())
    created_at=gettime()
    cursor=conn.cursor()
    cursor.execute("INSERT INTO KnowledgeBase (id,name,created_at,updated_at,document_list_id,file_count) VALUES (?,?,?,?,?,?)",(id,request.name,created_at,created_at,documentlistid,0))

    cursor.execute("INSERT INTO DocumentList (id,name,created_at,updated_at,knowledge_base_id) VALUES (?,?,?,?,?)",
                   (documentlistid, request.name,created_at,created_at,id))
    conn.commit()
    cursor.close()
    return Response(status="success",message="创建知识库成功")
@app.get("/getknowledgebaselist")
async def get_knowledgebase():
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM KnowledgeBase")
        result = cursor.fetchall()
        cursor.close()

        # 将查询结果转为字典
        knowledgebase = [
            {"id": row[0], "name": row[1], "created_at": row[2], "updated_at": row[3], "document_list_id": row[4],"file_count ":row[5]} for
            row in result]

        # 返回成功的响应
        return JSONResponse(content={"status": "success", "data": knowledgebase})

    except Exception as e:
        # 捕获异常并返回失败的响应
        return JSONResponse(status_code=500,
                            content={"status": "failed", "message": f"Error fetching knowledge base: {str(e)}"})
@app.post("/getDocumentlist")
async def get_documentlist(request:GetdocumentRequest):
    try:
        cursor = conn.cursor()

        # 获取 DocumentList id
        cursor.execute("SELECT id FROM DocumentList WHERE knowledge_base_id=?", (request.knowledge_base_id,))
        documentListid = cursor.fetchone()
        if not documentListid:
            cursor.close()
            return {"status": "failed", "message": "DocumentList not found"}
        documentListid = documentListid[0]

        # 计算分页偏移量
        offset = (request.page - 1) * request.page_size

        # 获取文件总数和文件列表
        cursor.execute('''
            SELECT COUNT(*), f.id, f.name, f.parsing_status, f.upload_at
            FROM DocumentList_File df
            JOIN File f ON f.id = df.file_id
            WHERE df.document_list_id = ?
            GROUP BY f.id
            LIMIT ? OFFSET ?
            ''', (documentListid, request.page_size, offset))

        # 获取总数和文件数据
        result = cursor.fetchall()

        if not result:
            cursor.close()
            return {"status": "failed", "message": "No files found"}

        total = result[0][0]  # 总文件数
        fileList = [{"id": row[1], "name": row[2], "parsing_status": row[3], "upload_at": row[4]} for row in result]

        cursor.close()

        return {"status": "success", "data": {"total": total, "fileList": fileList}}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"status": "failed", "message": f"Error fetching knowledge base: {str(e)}"})



@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    #用户输入问题
    cursor=conn.cursor()

    if request.conversation_id=="":
        request.conversation_id=str(uuid.uuid4())
        print(request.conversation_id)
        created_at=gettime()
        cursor.execute("INSERT INTO Conversations (id,user_id,title,created_at,updated_at,knowledge_base_id) VALUES (?,NULL,?,?,?,?)",(request.conversation_id,request.query,created_at,created_at,request.knowledge_base_id))
    message_id = str(uuid.uuid4())
    cursor.execute("INSERT INTO Messages (message_id,conversation_id,sender,content,mode,created_at)VALUES(?,?,?,?,?,?)",(message_id,request.conversation_id,"user",request.query,request.mode,gettime()))
    conn.commit()
    cursor.close()
    changrag(request.conversation_id)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode, only_need_context=request.only_need_context
                ),
            ),
        )
        result=result.strip("'''").replace("markdown","").strip()
        message_id=str(uuid.uuid4())
        #回答问题
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Messages (message_id,conversation_id,sender,content,mode,created_at)VALUES(?,?,?,?,?,?)",
                       (message_id,request.conversation_id, "ai", result, request.mode, gettime()))
        conn.commit()
        cursor.close()
        return Response(status="success", data=result,message=request.conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
# async def query_content2tree()
@app.get("/getfilelist")
async def getfilelist():
    cursor=conn.cursor()
    cursor.execute("SELECT id,name,embeded FROM File")
    filelist=cursor.fetchall()
    cursor.close()
    # 格式化查询结果
    files = [{"id": row[0], "name": row[1], "embeded": row[2]} for row in filelist]

    # 返回 JSON 响应
    return JSONResponse(content={"status": "success", "data": files})
@app.get("/gettasklist")
async def gettasklist():
    cursor=conn.cursor()
    cursor.execute("SELECT * FROM Tasks")
    tasklist=cursor.fetchall()
    cursor.close()
    tasks=[{"taskname":row[1],"description":row[2],"status":row[3],"starttime":row[4],"endtime":row[5]} for row in tasklist]
    return JSONResponse(content={"status":"success","data":tasks})



@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    global rag
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/uploadfile/{knowledgebase_id}")
async def upload_file(knowledgebase_id: str, file: UploadFile = File(...)):
    cursor = conn.cursor()

    # 检查文件是否已经存在
    cursor.execute("SELECT * FROM File WHERE name = ?", (file.filename,))
    fileflag = 0 if cursor.fetchall() else 1

    if fileflag:
        try:
            # 生成保存文件的路径，使用 os.path.join 拼接路径
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            print(f"Saving file to: {file_path}")

            # 将文件保存到服务器
            with open(file_path, "wb") as f:
                f.write(await file.read())

            fileid = str(uuid.uuid4())
            uploadtime = gettime()

            # 如果有 knowledgebase_id，插入 DocumentList_File 关联表
            if knowledgebase_id:
                cursor1 = conn.cursor()
                cursor1.execute("SELECT document_list_id FROM KnowledgeBase WHERE id = ?", (knowledgebase_id,))
                document_list_id_result = cursor1.fetchall()
                cursor1.close()
                document_list_id = document_list_id_result[0][0]  # 提取第一个结果中的 document_list_id
                cursor.execute("INSERT INTO DocumentList_File (document_list_id, file_id) VALUES (?, ?)",
                                     (document_list_id, fileid))

            # 插入文件信息到 File 表
            cursor.execute("INSERT INTO File (id, name, file_path, parsing_status, upload_at) VALUES (?, ?, ?, -1, ?)",
                           (fileid, file.filename, file_path, uploadtime))

            # 提交事务
            conn.commit()

            cursor.close()
            return JSONResponse(status_code=200, content={"status": "success", "message": file_path})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    else:
        return JSONResponse(status_code=400, content={"status": "failed", "message": "File already exists."})

@app.post("/insert_file", response_model=Response)
async def insert_file(file: InsertFileRequest):
    global rag
    try:
        filename=file.filename
        cursor=conn.cursor()
        cursor.execute("SELECT path FROM File WHERE name =?",(filename,))
        file_path=cursor.fetchall()
        cursor.close()
        print(file_path[0][0])
        if not os.path.exists(file_path[0][0]):
            raise HTTPException(
                status_code=404, detail=f"File not found: {file_path[0][0]}"
            )
        current_datetime = gettime()
        cursor=conn.cursor()
        cursor.execute('''
        INSERT INTO Tasks (taskname, description, status, starttime,endtime)
        VALUES ('Filembeded', ?, 'In Progress', ?,NULL)
        ''',(filename,current_datetime))
        conn.commit()
        cursor.close()
        # Read file content
        # try:
        #     with open(file_path[0][0], "r", encoding="utf-8") as f:
        #         content = f.read()
        # except UnicodeDecodeError:
        #     with open(file_path[0][0], "r", encoding="gbk") as f:
        #         content = f.read()
        print('开始读取')
        content= textract.read_text_file(file_path[0][0])
        # Insert file content into RAG system
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))
        current_datetime = gettime()
        cursor=conn.cursor()
        cursor.execute("UPDATE File SET embeded=? WHERE name=?",(1,filename))
        cursor.execute("UPDATE Tasks SET status=?,endtime=? WHERE description=?",("Completed",current_datetime,filename))
        conn.commit()
        cursor.close()

        return Response(
            status="success",
            message=f"File content from {file_path[0][0]} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
@app.post("/deleteentity")
async def delete_entity(request:DeleteEnetityRequest):
    global rag
    try:
        rag.delete_by_entity(request.enetityname)
        return Response(status="success",message=f"delete {request.enetityname} success",)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
@app.post("/deleteconversation")
async def delete_conversation(request:DeleteConversationRequest):
    try:
        cursor=conn.cursor()
        # 开始事务
        conn.execute('BEGIN TRANSACTION;')

        # 删除 message 表中与 conversation_id 相关的记录
        cursor.execute('DELETE FROM Messages WHERE conversation_id = ?', (request.conversationid,))

        # 删除 conversation 表中的记录
        cursor.execute('DELETE FROM Conversations WHERE id = ?', (request.conversationid,))

        # 提交事务
        conn.commit()
        cursor.close()
        return Response(status="success",
                        message=f"删除{request.conversationid}成功")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/visual")
async def visual_html():
    #visual_with_html()
    return Response(status="success",message="html visualize success")
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def gettime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example:
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py
