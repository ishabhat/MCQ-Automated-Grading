
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt

from flask import Flask, render_template, request, redirect, url_for, flash, session
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import random

ANSWER_KEY = {
    0: 1,
    1: 1,
    2: 0,
    3: 2,
    4: 1
}




app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'
app.config['SECRET_KEY']='thisisassecretkey'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)



login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

with app.app_context():
    db.create_all()


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():

    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # Check if the file has a valid extension (e.g., .png, .jpg)
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        if file:
            # Save the uploaded file temporarily
            filename = 'uploaded_image.png'
            file.save(filename)

            # Process the uploaded image
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Image processing code
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)

            # Find contours in the edge map
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            docCnt = None

            if len(cnts) > 0:
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                    if len(approx) == 4:
                        docCnt = approx
                        break

            if docCnt is not None:
                # Apply a four-point perspective transform
                paper = four_point_transform(image, docCnt.reshape(4, 2))
                warped = four_point_transform(gray, docCnt.reshape(4, 2))

                # Apply Otsu's thresholding method
                thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                # Find contours in the thresholded image
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                questionCnts = []

                # Loop over the contours
                for c in cnts:
                    (x, y, w, h) = cv2.boundingRect(c)
                    ar = w / float(h)

                    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
                        questionCnts.append(c)

                # Sort the question contours top-to-bottom
                questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
                correct = 0

                # Loop over the questions
                for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
                    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
                    bubbled = None

                    # Loop over the sorted contours
                    for (j, c) in enumerate(cnts):
                        mask = np.zeros(thresh.shape, dtype="uint8")
                        cv2.drawContours(mask, [c], -1, 255, -1)

                        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                        total = cv2.countNonZero(mask)

                        if bubbled is None or total > bubbled[0]:
                            bubbled = (total, j)

                    color = (0, 0, 255)
                    k = ANSWER_KEY[q]

                    if k == bubbled[1]:
                        color = (0, 255, 0)
                        correct += 1

                    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

                # Calculate the score
                total_questions = len(ANSWER_KEY)
                
                percentage=(correct / total_questions) * 100
                print("[INFO] score: {:.2f}%".format(percentage))
                cv2.putText(paper,f"{correct} / {total_questions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.imshow("Original", image)
                cv2.imshow("Exam", paper)
                cv2.waitKey(0)
                cv2.putText(image, f"{correct} / {total_questions}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0),
                            2)

            # Save the processed image to a file
            result_image_filename = 'static/result_image.png'
            cv2.imwrite(result_image_filename, image)

            return render_template('dashboard.html', score=percentage, result_image=result_image_filename)

    return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)




if __name__ == '__main__':
    app.run(debug=True)
