import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import hashlib
import os

def setup_authentication():
    """Setup user authentication system"""
    
    # Load configuration
    if os.path.exists('config.yaml'):
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
    else:
        # Default config
        config = {
            'credentials': {
                'usernames': {
                    'demo': {
                        'email': 'demo@example.com',
                        'name': 'Demo User',
                        'password': stauth.Hasher(['demo123']).generate()[0]
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'neurosegment_auth',
                'name': 'neurosegment_cookie'
            }
        }
        
        with open('config.yaml', 'w') as file:
            yaml.dump(config, file)
    
    # Create authenticator object (remove preauthorized parameter)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
    
    return authenticator

def user_registration_form(authenticator):
    """Display user registration form"""
    try:
        # Add preauthorization parameter to register_user function
        if authenticator.register_user('Register user', preauthorization=False):
            st.success('User registered successfully')
            # Update config file
            with open('config.yaml', 'w') as file:
                yaml.dump(authenticator.config, file)
    except Exception as e:
        st.error(f"Error: {e}")

def check_admin(username):
    """Check if user has admin privileges"""
    # In a real application, you'd have a proper admin system
    return username == "admin"