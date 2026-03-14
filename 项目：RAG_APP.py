import streamlit as st
import tempfile
import os
from pathlib import Path

# LangChain 组件（注意用 langchain_community）
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain.chains import RetrievalQA
from openai import OpenAI

# 页面配置
st.set_page_config(page_title="RAG知识库问答", page_icon="📚")
st.title("📚 RAG知识库问答系统")

# 初始化session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏：文档上传
with st.sidebar:
    st.header("📤 上传知识库文档")
    uploaded_files = st.file_uploader(
        "选择PDF或TXT文件",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

    if st.button("处理文档", type="primary"):
        if uploaded_files:
            with st.spinner("正在处理文档..."):
                # 1. 加载所有文档
                documents = []
                for uploaded_file in uploaded_files:
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # 根据文件类型选择加载器
                    if uploaded_file.type == "application/pdf":
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding='utf-8')

                    documents.extend(loader.load())

                    # 删除临时文件
                    os.unlink(tmp_path)

                # 2. 文本切片
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,  # 每块大小
                    chunk_overlap=50,  # 重叠部分
                    length_function=len,
                    separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)

                # 3. 创建向量库
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",  # 轻量级模型
                    model_kwargs={'device': 'cpu'}
                )
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"  # 持久化存储
                )

                st.success(f"✅ 处理完成！共 {len(chunks)} 个段落")
        else:
            st.warning("请先上传文件")

# 主界面：对话
if st.session_state.vectorstore:
    # 显示对话历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入
    if prompt := st.chat_input("想问什么？"):
        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # AI回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                try:
                    # 1. 检索相关文档
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": 3}  # 返回最相关的3段
                    )
                    docs = retriever.get_relevant_documents(prompt)

                    # 2. 准备上下文
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # 3. 调用大模型
                    client = OpenAI(
                        api_key=st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv(
                            "OPENAI_API_KEY"),
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
                    )

                    # 构建prompt
                    system_prompt = f"""你是一个基于知识库回答问题的助手。
请根据以下资料回答问题。如果资料中没有相关信息，就说"知识库中没有找到相关信息"。

相关资料：
{context}
"""

                    response = client.chat.completions.create(
                        model="qwen-plus",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        stream=True
                    )

                    # 流式输出
                    full_response = ""
                    message_placeholder = st.empty()
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"出错了：{str(e)}")

else:
    st.info("👈 请在左侧上传文档开始使用")