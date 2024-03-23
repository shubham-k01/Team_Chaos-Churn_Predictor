import smtplib
import ssl
import streamlit as st
from email.message import EmailMessage

# Define email sender and receiver
email_sender = 'deepchaudhary2047@gmail.com'
email_password = 'girrzjjbttpdnuqv'
email_receiver = 'sarojomkar5@gmail.com'

# Set the subject of the email
# subject = 'Check out my new video!'

recipient_name = ""
recipient_pos = ""
recipient_con =""


# Define the HTML content for the email body



def sender(name,pos,contact):
    # Create an EmailMessage object and set its properties
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = "Early Warning against Customer Churn"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Account Termination Notice</title>
    </head>
    <body style="font-family: Arial, sans-serif;">

    <h2 style="color: #333;">Subject: Important Notice: Customer Account Termination</h2>

    <p>Dear {name},</p>

    <p>We hope this message finds you well. We are writing to inform you of an important development regarding one of our valued customers, [Customer's Name]. After careful consideration and discussions, [Customer's Name] has decided to terminate their services with our company.</p>

    <p>While we respect their decision, we also want to take this opportunity to express our gratitude for the trust and support [Customer's Name] has shown us during their time with our services. We understand that this decision may have implications for our ongoing partnership and wish to ensure a smooth transition process.</p>

    <p>As part of our commitment to maintaining positive relationships with all our clients, we are prepared to assist in any way possible to facilitate the transition and address any outstanding matters. Our team is available to discuss the necessary steps and provide support as needed to minimize any disruptions and ensure a seamless experience for both parties.</p>

    <p>We value the feedback and insights of our customers, including those who choose to move on, as it helps us continuously improve our services and meet the evolving needs of our clients.</p>

    <p>Should you have any questions, require additional information, or wish to discuss this matter further, please do not hesitate to reach out to us. We look forward to continuing our collaboration with your esteemed company and are committed to delivering exceptional service and value.</p>

    <p>Thank you for your attention to this matter.</p>

    <p>Best regards,<br>
    {name}<br>
    {pos}<br>
    {contact}</p>

    </body>
    </html>
    """
    
    em.set_content(html_content, subtype='html')  # Set the content type to HTML

    # Add SSL (layer of security)
    context = ssl.create_default_context()

    # Log in and send the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        # smtp.login(email_sender, email_password)
        smtp.login(email_sender, st.secrets['PASS'])
        smtp.send_message(em)