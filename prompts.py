from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
)

system_prompt = (
            '''
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you dont know. Use three sentences maximum and keep the answer concise.
            
            {context}
            
            '''
)

qa_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', system_prompt),
                ('human', '{input}')
            ]
        )

contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ('system', contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name='history'),
                ('human', '{input}')
            ]
        )