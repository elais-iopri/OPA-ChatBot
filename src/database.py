import streamlit as st
import datetime
from typing import Dict
from sqlalchemy import text

conn = st.connection('mysql', type='sql')

def save_chat_history(chat_history : Dict, conversation_data : Dict) -> Dict:
    try:
        with conn.session as session:

            # parsing data
            data = {
                "chat_id" : chat_history["chat_id"],
                "message_user" : chat_history["chat_messages"]["user"],
                "message_assistant" : chat_history["chat_messages"]["assistant"],
                "chat_histories_id" : conversation_data["id"],
                "conversation_id" : conversation_data["conversation_id"],
                "previous_chat_id" : chat_history["previous_chat_id"]
            }

            stmt = text("INSERT INTO feedback_chat.feedback(chat_id, chat_histories_id, conversation_id, message_user, message_assistant, previous_chat_id) VALUES (:chat_id, :chat_histories_id, :conversation_id, :message_user, :message_assistant, :previous_chat_id)")

            result = session.execute(stmt, params=data)

            inserted_id = result.lastrowid

            stmt2 = text("SELECT * FROM feedback_chat.feedback WHERE id = :id")
            
            results_row = session.execute(stmt2, {"id" : inserted_id}).mappings().fetchone()

            return dict(results_row)

    except Exception as e:
        st.write(f"ERROR save_chat_history:\n{e}")

def get_user_by_id (user_id : int) -> Dict:
    try:
        with conn.session as session:

            stmt = text("SELECT id, ip_client_id, session_id FROM feedback_chat.users WHERE users.id = :user_id")
            result = session.execute(stmt, params={"user_id": user_id}).mappings().fetchone()
        
        return dict(result)
    
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
        
        return dict(result)
    
    except Exception as e:
        st.write(e)
        return None


def save_ip_client(ip_address : str) -> int:
    try:
        data = {
            "ip_address" : ip_address,
            "first_seen" : datetime.datetime.now(tz=datetime.timezone.utc)
        }
        
        with conn.session as session:
            stml = text('INSERT INTO feedback_chat.ip_clients(ip_address, first_seen) VALUES (:ip_address, :first_seen)')
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
            "log_time" : datetime.datetime.now(tz=datetime.timezone.utc)
        }

        with conn.session as session:
            stml = text('INSERT INTO feedback_chat.ip_login_histories(ip_client_id, log_time) VALUES (:ip_client_id, :log_time)')
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
            "created_at" : datetime.datetime.now(tz=datetime.timezone.utc)
        }
        
        with conn.session as session:
            stml = text('INSERT INTO feedback_chat.users(ip_client_id, session_id, created_at) VALUES (:ip_client_id, :session_id, :created_at)')
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
            "conversation_id" : data["conversation_id"],
            "user_id" : data["user_id"]
        }
        
        with conn.session as session:
            stml = text('INSERT INTO feedback_chat.chat_histories(conversation_id, user_id) VALUES (:conversation_id, :user_id)')
            result = session.execute(stml, data)
            session.commit()

            inserted_id = result.lastrowid
            stmt2 = text('SELECT id, conversation_id, user_id FROM feedback_chat.chat_histories WHERE id = :id')
            result_row = session.execute(stmt2, {"id": inserted_id}).mappings().fetchone()
            
        return dict(result_row)

    except Exception as e:
        st.write(f"ERROR saving conversation:\n {e}")
        return 0
 