FROM ubuntu
RUN apt update
RUN apt install -y apache2 apache2-utils python3 libapache2-mod-wsgi-py3 python3-pip
WORKDIR /var/www/html
RUN rm ./*
ADD ./model ./model
ADD ./requirements.txt ./
ADD ./app.py ./
ADD ./app.wsgi ./
ADD ./templates ./templates
ADD ./static ./static
ADD ./utils.py ./
RUN pip3 install -r requirements.txt
ADD ./config.conf /etc/apache2/sites-available/000-default.conf
EXPOSE 80
CMD ["apache2ctl", "-D", "FOREGROUND"]

