import streamlit as st
import datetime
import html
import time
from typing import Dict
from sqlalchemy import text

conn = st.connection('chat_opa', type='sql')

def save_chat_history(chat_history : Dict, conversation_data : Dict) -> Dict:
    try:
        with conn.session as session:

            # parsing data
            data = {
                "chat_id" : chat_history["chat_id"],
                "message_user" : chat_history["chat_messages"]["user"],
                "message_assistant" : chat_history["chat_messages"]["assistant"],
                "conversation_id" : conversation_data["id"],
                "conversation_session_id" : conversation_data["session_id"],
                "previous_chat_id" : chat_history["previous_chat_id"]
            }

            stmt = text("INSERT INTO chat_opa.chat_histories(chat_id, conversation_id, conversation_session_id, message_user, message_assistant, previous_chat_id) VALUES (:chat_id, :conversation_id, :conversation_session_id, :message_user, :message_assistant, :previous_chat_id)")

            result = session.execute(stmt, params=data)

            session.commit()

            inserted_id = result.lastrowid

            stmt2 = text("SELECT * FROM chat_opa.chat_histories WHERE id = :id")
            
            results_row = session.execute(stmt2, {"id" : inserted_id}).mappings().fetchone()

        if results_row is not None:
            return dict(results_row)
        return None

    except Exception as e:
        st.write(f"ERROR save_chat_history:\n{e}")

def get_user_by_id (user_id : int) -> Dict:
    try:
        with conn.session as session:

            stmt = text("SELECT id, ip_client_id, session_id FROM chat_opa.users WHERE users.id = :user_id")
            result = session.execute(stmt, params={"user_id": user_id}).mappings().fetchone()
        
        if result is not None:
            return dict(result)
        
        return None
    
    except Exception as e:
        st.write(f"ERROR getting user by id:\n{e}")
        return None

def check_ip_already_exists(ip_address: str) -> Dict:
    """
        Check if IP already exists in database
    """
    try:

        with conn.session as session:
            stmt = text("SELECT users.id, users.ip_client_id, users.session_id FROM ip_clients RIGHT JOIN users on ip_clients.id = users.ip_client_id WHERE ip_clients.ip_address = :ip_address")
            result = session.execute(stmt, params={"ip_address": ip_address}).mappings().fetchone()
        
        if result is not None:
            return dict(result)
        
        return None
    
    except Exception as e:
        st.write(e)
        return None


def save_ip_client(ip_address : str) -> int:
    try:
        data = {
            "ip_address" : ip_address
        }
        
        with conn.session as session:
            stml = text('INSERT INTO chat_opa.ip_clients(ip_address) VALUES (:ip_address)')
            result = session.execute(stml, data)
            session.commit()

        return result.lastrowid
            
    except Exception as e:
        st.write(f"ERROR saving ip client: {str(e)}")
        return 0
    
def save_ip_login_history(ip_client_id: str) -> int:
    try:
        data = {
            "ip_client_id" : ip_client_id,
        }

        with conn.session as session:
            stml = text('INSERT INTO chat_opa.ip_login_histories(ip_client_id) VALUES (:ip_client_id)')
            result = session.execute(stml, data)
            session.commit()

        return result.lastrowid
        
    except Exception as e:
        st.write(f"ERROR saving ip login history: {str(e)}")
        return 0

def save_user(ip_client_id : str, session_id : str) -> int:
    try :
        data = {
            "ip_client_id" : ip_client_id,
            "session_id" : session_id,
        }
        
        with conn.session as session:
            stml = text('INSERT INTO chat_opa.users(ip_client_id, session_id) VALUES (:ip_client_id, :session_id)')
            result = session.execute(stml, data)
            session.commit()

        return result.lastrowid
    
    except Exception as e:
        st.write(f"ERROR saving user: {str(e)}")
        return 0
    

def save_new_user(ip_client : str, session_id : str) -> int:
    try:
        # Adding new IP client
        ip_client_id_last_row = save_ip_client(ip_client)

        # Update log ip client
        save_ip_login_history(ip_client_id_last_row)

        # Save new user
        return save_user(ip_client_id_last_row, session_id)

    except Exception as e:
        st.write(f"ERROR saving new user: {str(e)}")
        return 0
        
def save_conversation(data : Dict) -> int:
    try:
        data = {
            "session_id" : data["conversation_id"],
            "user_id" : data["user_id"]
        }
        
        with conn.session as session:
            stml = text('INSERT INTO chat_opa.conversations(session_id, user_id) VALUES (:session_id, :user_id)')
            result = session.execute(stml, data)
            session.commit()

            inserted_id = result.lastrowid
            stmt2 = text('SELECT id, session_id, user_id FROM chat_opa.conversations WHERE id = :id')
            result_row = session.execute(stmt2, {"id": inserted_id}).mappings().fetchone()
            
        if result_row is not None:
            return dict(result_row)
        
        return None

    except Exception as e:
        st.write(f"ERROR saving conversation:\n {e}")
        return 0


def dialog_chat_opa_on_change(index):       
    st.session_state._temp_feedback = st.session_state[f"text_area_{index}"]

@st.dialog("Feedback")
def give_chat_opa_dialog(chat_id, index):
    st.write(f"Anda memberikan 👎 dari respon chatbot")

    feedback = st.text_area(
            label="Beritahu kami",
            placeholder="Tuliskan alasan kenapa memberikan feedback 👎",
            key=f"text_area_{index}",
            on_change=dialog_chat_opa_on_change,
            args = [index]
        )
    
    if st.button("Submit"):
        if st.session_state._temp_feedback is not None and len(st.session_state._temp_feedback.strip()) > 0:
            # st.write(chat_id)
            # # st.write(st.session_state._temp_feedback)
            save_text_chat_opa(chat_id, feedback)
            st.success("Terimakasih atas feedback anda", icon="👍")
            st.session_state._temp_feedback = None
            time.sleep(1)
            st.rerun()
        else :
            st.error("Silahkan isi feedback terlebih dahulu", icon="🚫")

def save_thumb_chat_feedback(index, id):
    with conn.session as session :

        stmt = text("UPDATE chat_opa.chat_histories SET thumb_score = :thumb WHERE id = :id")
        result = session.execute(stmt, params= {"thumb" : st.session_state[f"fb_{index}"], "id" : id})
        st.session_state.chat_histories[index]["thumb_score"] = st.session_state[f"fb_{index}"] # thumb score saved in session state
        session.commit()
        
    if st.session_state[f"fb_{index}"] == 0 :
        give_chat_opa_dialog(id, index)


def save_text_chat_opa(id, feedback) :
    with conn.session as session:
        stmt = text("UPDATE chat_opa.chat_histories SET feedback = :feedback WHERE id = :id")
        result = session.execute(stmt, params={"id" : id, "feedback" : html.escape(feedback)})
        session.commit()

        if result.lastrowid > 0 :
            return True
        else :
            return False
