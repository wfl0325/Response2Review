# 知识图谱抽取流程图

```mermaid
graph TD
    subgraph 用户 (浏览器)
        A[1. 用户选择文件<br>并点击“提交抽取”] --> B{2. 前端发送POST请求<br>到 /extract};
    end

    subgraph 后端 (Flask App)
        B --> C{3. /extract 路由<br>接收请求};
        C --> D[4. 获取数据库驱动<br>get_db_driver()];
        D --> E{5. 数据库连接<br>是否成功?};
        E -- 否 --> F[返回数据库连接错误];
        E -- 是 --> G[6. 获取上传的文件列表];
        G --> H[7. 遍历每个文件];
        H --> I{8. 文件类型<br>是否合法?};
        I -- 否 --> J[记录错误<br>处理下一个文件];
        I -- 是 --> K[9. 读取并解析文件内容<br>(TextProcessor)];
        
        subgraph 核心抽取逻辑 (KGExtract)
            K --> L[10. 实例化 ExtractAgent<br>(LLM客户端)];
            L --> M[11. 构建Prompt<br>(指令 + 文本内容)];
            M --> N[12. 调用LLM进行<br>信息抽取];
        end

        subgraph AI服务 (大语言模型)
            N -- 发送请求 --> O[13. LLM处理文本<br>并按格式返回三元组];
            O -- 返回结果 --> P[14. 后端接收LLM<br>返回的文本结果];
        end

        subgraph 数据入库
            P --> Q[15. 逐行解析<br>返回的三元组];
            Q --> R[16. 清洗数据<br>(sanitize_for_neo4j)];
            R --> S[17. 构造Cypher MERGE<br>查询语句];
            S -- 执行查询 --> T((18. Neo4j数据库));
            T -- 写入/更新 --> Q;
        end
        J --> U[19. 收集所有<br>文件的处理结果];
        Q -- 处理完所有三元组 --> U;

        U --> V[20. 将处理结果<br>打包成JSON格式];
    end

    subgraph 用户 (浏览器)
        F --> W[21. 前端显示<br>错误提示];
        V --> X[21. 前端显示<br>成功结果与提示];
    end