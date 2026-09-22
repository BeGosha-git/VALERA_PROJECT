import os
import random
from flask import Blueprint, render_template, redirect, url_for, flash, request, session, current_app
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    # If user is already logged in, redirect to main page
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.get_by_username(username)
        if user and user.check_password(password):
            # Remember user preference (if checkbox selected)
            remember = 'remember' in request.form
            login_user(user, remember=remember)
            
            # Get next page from request args
            next_page = request.args.get('next')
            
            # Redirect to next page or main index
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(next_page or url_for('main.index'))
        else:
            flash('Invalid username or password', 'error')
    
    # Get random background image if enabled
    background_image = None
    if current_app.config.get('LOGIN_USE_BACKGROUND_IMAGES', False):
        bg_dir = os.path.join(current_app.static_folder, 'images', 'backgrounds')
        if os.path.isdir(bg_dir):
            # List all image files (jpg, jpeg, png, webp)
            image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
            images = [f for f in os.listdir(bg_dir) 
                     if os.path.isfile(os.path.join(bg_dir, f)) and 
                     f.lower().endswith(image_extensions)]
            
            if images:
                # Select a random image
                random_image = random.choice(images)
                background_image = f'images/backgrounds/{random_image}'
    
    return render_template('auth/login.html', background_image=background_image)

@auth_bp.route('/logout')
@login_required
def logout():
    """Logout route"""
    # Clear the session and logout user
    session = {}
    logout_user()
    flash('You have been logged out successfully', 'info')
    return redirect(url_for('auth.login'))