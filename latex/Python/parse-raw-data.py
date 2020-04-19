# BeautifulSoup needed to parse downloaded html files
from bs4 import BeautifulSoup

# glob and os are used to get the list of html files
import glob
import os

# MySQLdb is needed to write the data to the database
import MySQLdb as mdb
import sys

# Variables for the anonymous question id, user id and asker id
q_id = 1
u_id = 1
a_id = 1

q_id_str = ''
u_id_str = ''
a_id_str = ''

# Distionary to hold anonymous ask ids
ask_ids = {'no_ask_id': '9999'}

# Set the folder where the html files are
os.chdir("R:\\Raw")

# Get the list of files
for file in glob.glob("*.htm"):

  # Reset variables for the anonymous question id
  q_id = 1

  q_id_str = ''
  u_id_str = ''
  a_id_str = ''

  # Parse the html file using the html parser
  soup = BeautifulSoup(open(file), 'html.parser')

  # Connect to the database for each html file
  try:
    con = mdb.connect(
      'localhost',
      'root',
      'mysqlroot',
      'thesis',
      use_unicode=True,
      charset="utf8")

    cur = con.cursor()

    # Find each occurrence of the question box and process
    questions = soup.findAll("div", {"class": "questionBox"})

    for a_question in soup.findAll("div",
                     {"class": "questionBox"}):

      # Generate the anonymous user id and question id
      u_id_str = str(u_id).zfill(3)
      q_id_str = u_id_str + str(q_id).zfill(5)

      # These elements / attributes will always exist
      question_id = a_question["id"].split("_")[2].strip()
      user_id = a_question.find(
        attrs="reportFlagBox").a["href"].split("/")[3].strip()
      question = a_question.find(
        attrs="question").span.span.get_text().replace(
        "\n",
        "").strip()
      answer = a_question.find(
        attrs="answer").get_text().replace(
        "\n",
        "").strip()
      ask_time = a_question.find(
          attrs="time").a.get_text().strip()

      # These elements / attributes may or may not exist
      try:
        if a_question.find(
          attrs="author nowrap").a["href"].split("/")[3]:
          ask_id = a_question.find(
            attrs="author nowrap").a["href"].split(
                "/")[3].strip()
      except (AttributeError, KeyError):
        ask_id = "no_ask_id"

      try:
        if a_question.find(
          attrs="likeList people-like-block").a.get_text(
              ).split(" ")[0]:
          ask_likes = a_question.find(
            attrs="likeList people-like-block").a.get_text(
                ).split(" ")[0].strip()
      except (AttributeError, KeyError):
        ask_likes = ""

      # Anonymise the id of the asker
      if ask_ids.has_key(ask_id):
        a_id_str = ask_ids[ask_id]
      else:
        a_id_str = str(a_id).zfill(3)
        a_id = a_id + 1
        ask_ids.update({ask_id: a_id_str})

      cur.execute(
        "INSERT INTO thesis.raw_data (question_id    \
                                    , q_id,user_id   \
                                    , u_id,ask_id    \
                                    , question       \
                                    , answer         \
                                    , ask_time       \
                                    , ask_likes      \
                                    , a_id)          \
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (question_id,
         q_id_str,
         user_id,
         u_id_str,
         ask_id,
         question,
         answer,
         ask_time,
         ask_likes,
         a_id_str))

      # Increase question number
      q_id = q_id + 1

    con.commit()

  # Handle database errors
  except mdb.Error as e:

    print "Error %d: %s" % (e.args[0], e.args[1])
    sys.exit(1)

  # Close the database connection
  finally:
    if con:
      con.close()

  u_id = u_id + 1
