"""SQL fixtures for the offline demo corpus.

Each entry is a real, hand-authored SQLite schema + INSERT script plus the
Spider-style (question, gold SQL) pairs that go with it. The databases are
genuinely relational (foreign keys, NULLs, date strings, aggregates that matter)
so that execution accuracy on them is meaningful: a model that emits the wrong
JOIN or the wrong GROUP BY really does get a different result set.

The schemas intentionally mirror the flavour of Spider (academic, business,
geo, music, sports) so the generated prompts look like the real benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DbFixture:
    db_id: str
    ddl: str                      # CREATE TABLE ... ; INSERT ... ;
    examples: list[tuple[str, str]] = field(default_factory=list)  # (question, gold sql)


FIXTURES: list[DbFixture] = [
    # ------------------------------------------------------------------ #
    DbFixture(
        db_id="college",
        ddl="""
CREATE TABLE department (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    building TEXT,
    budget REAL
);
CREATE TABLE student (
    stu_id INTEGER PRIMARY KEY,
    stu_name TEXT NOT NULL,
    dept_id INTEGER,
    age INTEGER,
    gpa REAL,
    enroll_year INTEGER,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);
CREATE TABLE instructor (
    inst_id INTEGER PRIMARY KEY,
    inst_name TEXT NOT NULL,
    dept_id INTEGER,
    salary REAL,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);
CREATE TABLE course (
    course_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    dept_id INTEGER,
    credits INTEGER,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);
CREATE TABLE enrollment (
    enroll_id INTEGER PRIMARY KEY,
    stu_id INTEGER,
    course_id INTEGER,
    semester TEXT,
    grade REAL,
    FOREIGN KEY (stu_id) REFERENCES student(stu_id),
    FOREIGN KEY (course_id) REFERENCES course(course_id)
);
INSERT INTO department VALUES
 (1,'Computer Science','Turing Hall',4200000.0),
 (2,'Mathematics','Newton Hall',3100000.0),
 (3,'Physics','Curie Hall',2800000.0),
 (4,'History','Herodotus Hall',1500000.0),
 (5,'Biology','Darwin Hall',2600000.0);
INSERT INTO student VALUES
 (1,'Alice Chen',1,20,3.85,2021),
 (2,'Bob Ortiz',1,22,3.20,2019),
 (3,'Carol Nguyen',2,21,3.95,2020),
 (4,'David Kim',3,23,2.85,2018),
 (5,'Emma Rossi',2,19,3.60,2022),
 (6,'Frank Muller',4,24,2.40,2017),
 (7,'Grace Okafor',1,20,3.10,2021),
 (8,'Hiro Tanaka',5,22,3.75,2019),
 (9,'Ivy Petrova',3,21,3.45,2020),
 (10,'Jack Ryan',4,25,2.10,2016);
INSERT INTO instructor VALUES
 (1,'Alan Turing',1,145000.0),
 (2,'Ada Lovelace',1,138000.0),
 (3,'Emmy Noether',2,131000.0),
 (4,'Marie Curie',3,142000.0),
 (5,'Leonhard Euler',2,126000.0),
 (6,'Barbara McClintock',5,118000.0);
INSERT INTO course VALUES
 (1,'Intro to Programming',1,4),
 (2,'Data Structures',1,3),
 (3,'Machine Learning',1,3),
 (4,'Real Analysis',2,4),
 (5,'Linear Algebra',2,3),
 (6,'Quantum Mechanics',3,4),
 (7,'World History',4,3),
 (8,'Genetics',5,4),
 (9,'Databases',1,3);
INSERT INTO enrollment VALUES
 (1,1,1,'Fall2021',3.9),(2,1,2,'Spring2022',3.8),(3,2,1,'Fall2019',3.1),
 (4,2,2,'Spring2020',3.3),(5,3,4,'Fall2020',4.0),(6,3,5,'Spring2021',3.9),
 (7,4,6,'Fall2018',2.9),(8,5,4,'Fall2022',3.7),(9,5,5,'Spring2023',3.5),
 (10,6,7,'Fall2017',2.5),(11,7,1,'Fall2021',3.0),(12,7,9,'Spring2022',3.2),
 (13,8,8,'Fall2019',3.8),(14,9,6,'Fall2020',3.4),(15,9,5,'Spring2021',3.5),
 (16,10,7,'Fall2016',2.2),(17,1,9,'Fall2022',3.6),(18,3,3,'Fall2022',3.9);
""",
        examples=[
            ("How many students are enrolled in the Computer Science department?",
             "SELECT COUNT(*) FROM student WHERE dept_id = 1"),
            ("List the names of all students whose GPA is above 3.5.",
             "SELECT stu_name FROM student WHERE gpa > 3.5"),
            ("What is the average salary of instructors in the Mathematics department?",
             "SELECT AVG(salary) FROM instructor WHERE dept_id = 2"),
            ("Show the department name and budget of each department with a budget over 3 million.",
             "SELECT dept_name, budget FROM department WHERE budget > 3000000"),
            ("Find the names of students who have taken a course titled 'Databases'.",
             "SELECT T2.stu_name FROM enrollment AS T1 JOIN student AS T2 ON T1.stu_id = T2.stu_id "
             "JOIN course AS T3 ON T1.course_id = T3.course_id WHERE T3.title = 'Databases'"),
            ("How many courses does each department offer?",
             "SELECT dept_id, COUNT(*) FROM course GROUP BY dept_id"),
            ("Which instructor has the highest salary?",
             "SELECT inst_name FROM instructor ORDER BY salary DESC LIMIT 1"),
            ("List student names together with their department names.",
             "SELECT T1.stu_name, T2.dept_name FROM student AS T1 "
             "JOIN department AS T2 ON T1.dept_id = T2.dept_id"),
            ("What is the average GPA of students who enrolled in 2021 or later?",
             "SELECT AVG(gpa) FROM student WHERE enroll_year >= 2021"),
            ("Show each department name and the number of students in it, ordered by the count descending.",
             "SELECT T2.dept_name, COUNT(*) FROM student AS T1 "
             "JOIN department AS T2 ON T1.dept_id = T2.dept_id "
             "GROUP BY T2.dept_name ORDER BY COUNT(*) DESC"),
            ("Find the titles of courses that have never been enrolled in.",
             "SELECT title FROM course WHERE course_id NOT IN (SELECT course_id FROM enrollment)"),
            ("List the names of instructors whose salary is above the average instructor salary.",
             "SELECT inst_name FROM instructor WHERE salary > (SELECT AVG(salary) FROM instructor)"),
            ("How many distinct students have taken at least one course in the Physics department?",
             "SELECT COUNT(DISTINCT T1.stu_id) FROM enrollment AS T1 "
             "JOIN course AS T2 ON T1.course_id = T2.course_id WHERE T2.dept_id = 3"),
            ("Show the building and total budget for each building, sorted by total budget descending.",
             "SELECT building, SUM(budget) FROM department GROUP BY building ORDER BY SUM(budget) DESC"),
            ("What is the name of the oldest student?",
             "SELECT stu_name FROM student ORDER BY age DESC LIMIT 1"),
            ("Find courses whose credits are greater than 3 and that belong to the Computer Science department.",
             "SELECT title FROM course WHERE credits > 3 AND dept_id = 1"),
            ("For each student, show the student name and their average grade.",
             "SELECT T1.stu_name, AVG(T2.grade) FROM student AS T1 "
             "JOIN enrollment AS T2 ON T1.stu_id = T2.stu_id GROUP BY T1.stu_name"),
            ("Which departments have more than 2 students?",
             "SELECT T2.dept_name FROM student AS T1 JOIN department AS T2 ON T1.dept_id = T2.dept_id "
             "GROUP BY T2.dept_name HAVING COUNT(*) > 2"),
            ("List the titles of courses taken by the student named Alice Chen.",
             "SELECT T3.title FROM student AS T1 JOIN enrollment AS T2 ON T1.stu_id = T2.stu_id "
             "JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T1.stu_name = 'Alice Chen'"),
            ("Show the total number of enrollments per semester.",
             "SELECT semester, COUNT(*) FROM enrollment GROUP BY semester"),
            ("What is the maximum GPA among students in the History department?",
             "SELECT MAX(gpa) FROM student WHERE dept_id = 4"),
            ("Find the names of students who are enrolled in both 'Data Structures' and 'Databases'.",
             "SELECT T1.stu_name FROM student AS T1 "
             "JOIN enrollment AS T2 ON T1.stu_id = T2.stu_id "
             "JOIN course AS T3 ON T2.course_id = T3.course_id "
             "WHERE T3.title = 'Data Structures' INTERSECT "
             "SELECT T1.stu_name FROM student AS T1 "
             "JOIN enrollment AS T2 ON T1.stu_id = T2.stu_id "
             "JOIN course AS T3 ON T2.course_id = T3.course_id WHERE T3.title = 'Databases'"),
        ],
    ),
    # ------------------------------------------------------------------ #
    DbFixture(
        db_id="restaurant",
        ddl="""
CREATE TABLE restaurant (
    rest_id INTEGER PRIMARY KEY,
    rest_name TEXT NOT NULL,
    city TEXT,
    rating REAL,
    opened_year INTEGER
);
CREATE TABLE dish (
    dish_id INTEGER PRIMARY KEY,
    dish_name TEXT NOT NULL,
    rest_id INTEGER,
    price REAL,
    category TEXT,
    FOREIGN KEY (rest_id) REFERENCES restaurant(rest_id)
);
CREATE TABLE review (
    review_id INTEGER PRIMARY KEY,
    rest_id INTEGER,
    reviewer TEXT,
    stars INTEGER,
    review_date TEXT,
    FOREIGN KEY (rest_id) REFERENCES restaurant(rest_id)
);
CREATE TABLE chef (
    chef_id INTEGER PRIMARY KEY,
    chef_name TEXT NOT NULL,
    rest_id INTEGER,
    specialty TEXT,
    years_experience INTEGER,
    FOREIGN KEY (rest_id) REFERENCES restaurant(rest_id)
);
INSERT INTO restaurant VALUES
 (1,'Golden Dragon','Beijing',4.6,2008),
 (2,'La Piazza','Rome',4.2,1995),
 (3,'Sakura House','Tokyo',4.8,2012),
 (4,'Taco Loco','Mexico City',3.9,2015),
 (5,'Curry Palace','Delhi',4.4,2003),
 (6,'Blue Harbor','Seattle',4.1,2018),
 (7,'Steak & Stone','Austin',3.6,2010);
INSERT INTO dish VALUES
 (1,'Peking Duck',1,38.0,'main'),(2,'Kung Pao Chicken',1,22.0,'main'),
 (3,'Dumplings',1,12.0,'starter'),(4,'Margherita Pizza',2,18.0,'main'),
 (5,'Tiramisu',2,9.0,'dessert'),(6,'Sushi Platter',3,45.0,'main'),
 (7,'Miso Soup',3,6.0,'starter'),(8,'Tempura',3,16.0,'main'),
 (9,'Tacos al Pastor',4,11.0,'main'),(10,'Guacamole',4,8.0,'starter'),
 (11,'Butter Chicken',5,20.0,'main'),(12,'Naan',5,4.0,'starter'),
 (13,'Clam Chowder',6,14.0,'starter'),(14,'Grilled Salmon',6,29.0,'main'),
 (15,'Ribeye Steak',7,42.0,'main'),(16,'Caesar Salad',7,13.0,'starter'),
 (17,'Fried Rice',1,14.0,'main'),(18,'Gelato',2,7.0,'dessert');
INSERT INTO review VALUES
 (1,1,'Tina',5,'2023-01-15'),(2,1,'Marco',4,'2023-02-20'),(3,2,'Sofia',4,'2023-01-08'),
 (4,2,'Liam',5,'2023-03-11'),(5,3,'Yuki',5,'2023-02-02'),(6,3,'Anna',5,'2023-04-19'),
 (7,4,'Carlos',3,'2023-01-25'),(8,4,'Nina',4,'2023-03-30'),(9,5,'Raj',4,'2023-02-14'),
 (10,5,'Priya',5,'2023-04-05'),(11,6,'Owen',4,'2023-03-22'),(12,6,'Mia',4,'2023-05-01'),
 (13,7,'Hank',3,'2023-02-18'),(14,7,'Zoe',2,'2023-04-27'),(15,1,'Peter',5,'2023-05-09');
INSERT INTO chef VALUES
 (1,'Wei Zhang',1,'Sichuan',18),
 (2,'Giovanni Russo',2,'Italian',25),
 (3,'Kenji Sato',3,'Japanese',15),
 (4,'Maria Gonzalez',4,'Mexican',9),
 (5,'Arjun Mehta',5,'Indian',21),
 (6,'Sarah Blake',6,'Seafood',7),
 (7,'Tom Harker',7,'Steak',12),
 (8,'Liu Fang',1,'Dim Sum',11);
""",
        examples=[
            ("How many restaurants are there in each city?",
             "SELECT city, COUNT(*) FROM restaurant GROUP BY city"),
            ("List the names of restaurants with a rating above 4.5.",
             "SELECT rest_name FROM restaurant WHERE rating > 4.5"),
            ("What is the most expensive dish on the menu?",
             "SELECT dish_name FROM dish ORDER BY price DESC LIMIT 1"),
            ("Show the name and price of every dish at 'Golden Dragon'.",
             "SELECT T2.dish_name, T2.price FROM restaurant AS T1 "
             "JOIN dish AS T2 ON T1.rest_id = T2.rest_id WHERE T1.rest_name = 'Golden Dragon'"),
            ("What is the average price of dishes in the dessert category?",
             "SELECT AVG(price) FROM dish WHERE category = 'dessert'"),
            ("Which restaurant has the highest average review stars?",
             "SELECT T1.rest_name FROM restaurant AS T1 "
             "JOIN review AS T2 ON T1.rest_id = T2.rest_id "
             "GROUP BY T1.rest_name ORDER BY AVG(T2.stars) DESC LIMIT 1"),
            ("Find the names of chefs who have more than 15 years of experience.",
             "SELECT chef_name FROM chef WHERE years_experience > 15"),
            ("List each restaurant name together with its number of reviews.",
             "SELECT T1.rest_name, COUNT(*) FROM restaurant AS T1 "
             "JOIN review AS T2 ON T1.rest_id = T2.rest_id GROUP BY T1.rest_name"),
            ("Show all reviews with 5 stars along with the restaurant name.",
             "SELECT T2.rest_name, T1.reviewer FROM review AS T1 "
             "JOIN restaurant AS T2 ON T1.rest_id = T2.rest_id WHERE T1.stars = 5"),
            ("What is the total revenue if every dish were sold once, per restaurant?",
             "SELECT T1.rest_name, SUM(T2.price) FROM restaurant AS T1 "
             "JOIN dish AS T2 ON T1.rest_id = T2.rest_id GROUP BY T1.rest_name"),
            ("Which cities have restaurants that opened before 2010?",
             "SELECT DISTINCT city FROM restaurant WHERE opened_year < 2010"),
            ("Find the names of restaurants that have no reviews.",
             "SELECT rest_name FROM restaurant WHERE rest_id NOT IN (SELECT rest_id FROM review)"),
            ("How many dishes are in the 'starter' category?",
             "SELECT COUNT(*) FROM dish WHERE category = 'starter'"),
            ("List the restaurant name and its chef's specialty.",
             "SELECT T1.rest_name, T2.specialty FROM restaurant AS T1 "
             "JOIN chef AS T2 ON T1.rest_id = T2.rest_id"),
            ("What is the lowest rating among restaurants in the USA?",
             "SELECT MIN(rating) FROM restaurant WHERE city IN ('Seattle','Austin')"),
            ("Show the average number of stars per restaurant, ordered from highest to lowest.",
             "SELECT rest_id, AVG(stars) FROM review GROUP BY rest_id ORDER BY AVG(stars) DESC"),
            ("Find restaurants that serve both a main dish and a dessert.",
             "SELECT T1.rest_name FROM restaurant AS T1 "
             "JOIN dish AS T2 ON T1.rest_id = T2.rest_id WHERE T2.category = 'main' "
             "INTERSECT SELECT T1.rest_name FROM restaurant AS T1 "
             "JOIN dish AS T2 ON T1.rest_id = T2.rest_id WHERE T2.category = 'dessert'"),
            ("Which restaurant has the most dishes?",
             "SELECT T1.rest_name FROM restaurant AS T1 JOIN dish AS T2 ON T1.rest_id = T2.rest_id "
             "GROUP BY T1.rest_name ORDER BY COUNT(*) DESC LIMIT 1"),
            ("List the names of restaurants whose average dish price is above 20.",
             "SELECT T1.rest_name FROM restaurant AS T1 JOIN dish AS T2 ON T1.rest_id = T2.rest_id "
             "GROUP BY T1.rest_name HAVING AVG(T2.price) > 20"),
            ("How many reviews were written in 2023?",
             "SELECT COUNT(*) FROM review WHERE review_date LIKE '2023%'"),
        ],
    ),
    # ------------------------------------------------------------------ #
    DbFixture(
        db_id="flight",
        ddl="""
CREATE TABLE airline (
    airline_id INTEGER PRIMARY KEY,
    airline_name TEXT NOT NULL,
    country TEXT,
    founded_year INTEGER
);
CREATE TABLE airport (
    airport_id INTEGER PRIMARY KEY,
    airport_code TEXT NOT NULL,
    city TEXT,
    country TEXT
);
CREATE TABLE flight (
    flight_id INTEGER PRIMARY KEY,
    airline_id INTEGER,
    origin_id INTEGER,
    dest_id INTEGER,
    departure_time TEXT,
    duration_min INTEGER,
    price REAL,
    seats_total INTEGER,
    seats_sold INTEGER,
    FOREIGN KEY (airline_id) REFERENCES airline(airline_id),
    FOREIGN KEY (origin_id) REFERENCES airport(airport_id),
    FOREIGN KEY (dest_id) REFERENCES airport(airport_id)
);
CREATE TABLE passenger (
    passenger_id INTEGER PRIMARY KEY,
    passenger_name TEXT NOT NULL,
    age INTEGER,
    flight_id INTEGER,
    seat_class TEXT,
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id)
);
INSERT INTO airline VALUES
 (1,'Air China','China',1988),
 (2,'Lufthansa','Germany',1953),
 (3,'Delta','USA',1925),
 (4,'Emirates','UAE',1985),
 (5,'ANA','Japan',1952);
INSERT INTO airport VALUES
 (1,'PEK','Beijing','China'),
 (2,'FRA','Frankfurt','Germany'),
 (3,'JFK','New York','USA'),
 (4,'DXB','Dubai','UAE'),
 (5,'HND','Tokyo','Japan'),
 (6,'PVG','Shanghai','China');
INSERT INTO flight VALUES
 (1,1,1,2,'08:30',615,780.0,300,281),
 (2,1,1,3,'13:45',840,950.0,280,240),
 (3,2,2,1,'10:15',600,820.0,290,265),
 (4,2,2,4,'22:00',390,640.0,320,310),
 (5,3,3,4,'18:20',780,890.0,260,198),
 (6,4,4,5,'02:10',560,720.0,340,322),
 (7,5,5,1,'09:40',230,410.0,200,176),
 (8,1,6,4,'11:55',540,700.0,300,288),
 (9,3,3,1,'16:30',850,1020.0,250,190),
 (10,5,5,3,'07:05',740,910.0,220,201),
 (11,4,4,2,'12:40',420,560.0,330,300),
 (12,2,2,5,'20:15',700,860.0,280,233);
INSERT INTO passenger VALUES
 (1,'Lily Wang',28,1,'economy'),(2,'Tom Becker',45,1,'business'),
 (3,'Sara Lee',33,2,'economy'),(4,'Ivan Petrov',51,2,'first'),
 (5,'Maya Patel',24,3,'economy'),(6,'Jonas Weber',39,3,'business'),
 (7,'Chen Yu',30,4,'economy'),(8,'Aisha Khan',27,4,'economy'),
 (9,'Mark Davis',58,5,'business'),(10,'Elena Rossi',35,5,'economy'),
 (11,'Hiro Suzuki',41,6,'first'),(12,'Nina Berg',22,6,'economy'),
 (13,'Omar Ali',46,7,'business'),(14,'Kate Brown',31,7,'economy'),
 (15,'Ravi Kumar',29,8,'economy'),(16,'Sophie Martin',37,8,'business'),
 (17,'Diego Lopez',26,9,'economy'),(18,'Anna Nowak',44,9,'first'),
 (19,'Paul Green',53,10,'business'),(20,'Yuna Kim',21,10,'economy'),
 (21,'Lars Nilsson',48,11,'economy'),(22,'Fatima Noor',34,11,'business'),
 (23,'Ben Carter',25,12,'economy'),(24,'Chloe Dubois',40,12,'first');
""",
        examples=[
            ("How many flights does each airline operate?",
             "SELECT T1.airline_name, COUNT(*) FROM airline AS T1 "
             "JOIN flight AS T2 ON T1.airline_id = T2.airline_id GROUP BY T1.airline_name"),
            ("List the codes of airports in China.",
             "SELECT airport_code FROM airport WHERE country = 'China'"),
            ("What is the average ticket price of flights from Beijing to Frankfurt?",
             "SELECT AVG(T1.price) FROM flight AS T1 JOIN airport AS T2 ON T1.origin_id = T2.airport_id "
             "JOIN airport AS T3 ON T1.dest_id = T3.airport_id "
             "WHERE T2.city = 'Beijing' AND T3.city = 'Frankfurt'"),
            ("Find the names of passengers who flew in first class.",
             "SELECT passenger_name FROM passenger WHERE seat_class = 'first'"),
            ("Show the flight id and price for flights that are cheaper than 700.",
             "SELECT flight_id, price FROM flight WHERE price < 700"),
            ("Which airline was founded the earliest?",
             "SELECT airline_name FROM airline ORDER BY founded_year ASC LIMIT 1"),
            ("How many seats are still available on each flight?",
             "SELECT flight_id, seats_total - seats_sold FROM flight"),
            ("List the names of passengers older than 40 who travelled in business class.",
             "SELECT passenger_name FROM passenger WHERE age > 40 AND seat_class = 'business'"),
            ("What is the total number of passengers carried by each airline?",
             "SELECT T1.airline_name, COUNT(*) FROM airline AS T1 "
             "JOIN flight AS T2 ON T1.airline_id = T2.airline_id "
             "JOIN passenger AS T3 ON T2.flight_id = T3.flight_id GROUP BY T1.airline_name"),
            ("Find the longest flight by duration.",
             "SELECT flight_id FROM flight ORDER BY duration_min DESC LIMIT 1"),
            ("Show each destination city and the number of flights arriving there.",
             "SELECT T2.city, COUNT(*) FROM flight AS T1 "
             "JOIN airport AS T2 ON T1.dest_id = T2.airport_id GROUP BY T2.city"),
            ("Which flights are operated by airlines from Germany?",
             "SELECT T1.flight_id FROM flight AS T1 JOIN airline AS T2 ON T1.airline_id = T2.airline_id "
             "WHERE T2.country = 'Germany'"),
            ("What is the average age of passengers on flight 1?",
             "SELECT AVG(age) FROM passenger WHERE flight_id = 1"),
            ("List airports that are not the origin of any flight.",
             "SELECT airport_code FROM airport WHERE airport_id NOT IN "
             "(SELECT origin_id FROM flight)"),
            ("Find the airline with the highest average ticket price.",
             "SELECT T1.airline_name FROM airline AS T1 JOIN flight AS T2 ON T1.airline_id = T2.airline_id "
             "GROUP BY T1.airline_name ORDER BY AVG(T2.price) DESC LIMIT 1"),
            ("How many flights depart from each airport? Show the airport code.",
             "SELECT T2.airport_code, COUNT(*) FROM flight AS T1 "
             "JOIN airport AS T2 ON T1.origin_id = T2.airport_id GROUP BY T2.airport_code"),
            ("Show the names of passengers who paid for a flight that costs more than 900.",
             "SELECT T3.passenger_name FROM flight AS T1 "
             "JOIN passenger AS T3 ON T1.flight_id = T3.flight_id WHERE T1.price > 900"),
            ("Which countries have more than one airport in the database?",
             "SELECT country FROM airport GROUP BY country HAVING COUNT(*) > 1"),
            ("What is the total number of seats sold across all flights?",
             "SELECT SUM(seats_sold) FROM flight"),
            ("List flights that have sold more than 90 percent of their seats.",
             "SELECT flight_id FROM flight WHERE seats_sold * 1.0 / seats_total > 0.9"),
        ],
    ),
    # ------------------------------------------------------------------ #
    DbFixture(
        db_id="music",
        ddl="""
CREATE TABLE artist (
    artist_id INTEGER PRIMARY KEY,
    artist_name TEXT NOT NULL,
    country TEXT,
    genre TEXT,
    debut_year INTEGER
);
CREATE TABLE album (
    album_id INTEGER PRIMARY KEY,
    album_title TEXT NOT NULL,
    artist_id INTEGER,
    release_year INTEGER,
    label TEXT,
    FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
);
CREATE TABLE song (
    song_id INTEGER PRIMARY KEY,
    song_title TEXT NOT NULL,
    album_id INTEGER,
    duration_sec INTEGER,
    plays INTEGER,
    FOREIGN KEY (album_id) REFERENCES album(album_id)
);
CREATE TABLE concert (
    concert_id INTEGER PRIMARY KEY,
    artist_id INTEGER,
    venue TEXT,
    city TEXT,
    concert_date TEXT,
    tickets_sold INTEGER,
    FOREIGN KEY (artist_id) REFERENCES artist(artist_id)
);
INSERT INTO artist VALUES
 (1,'Aurora Skies','Norway','indie',2011),
 (2,'The Velvet Keys','USA','rock',2005),
 (3,'Mira Sol','Spain','pop',2015),
 (4,'Kaito Ren','Japan','electronic',2009),
 (5,'Desert Bloom','Morocco','folk',2017),
 (6,'Neon Harbor','UK','synthpop',2013),
 (7,'Silver Pine','Canada','folk',2001);
INSERT INTO album VALUES
 (1,'Northern Lights',1,2014,'Fjord Records'),
 (2,'Midnight Sun',1,2017,'Fjord Records'),
 (3,'Iron Curtain',2,2008,'Hammer Label'),
 (4,'Glass Hearts',2,2012,'Hammer Label'),
 (5,'Verano',3,2016,'Sol Music'),
 (6,'Tokyo Circuit',4,2011,'Ren Audio'),
 (7,'Sahara Wind',5,2019,'Bloom Studio'),
 (8,'Electric Tide',6,2014,'Harbor Sound'),
 (9,'Quiet Rivers',7,2003,'Pine Records'),
 (10,'Lunar Phase',4,2018,'Ren Audio');
INSERT INTO song VALUES
 (1,'Frozen Lake',1,245,1520000),(2,'Aurora',1,312,2890000),
 (3,'Polar Night',2,198,940000),(4,'Ember',3,276,1730000),
 (5,'Anvil',3,321,1210000),(6,'Glass Heart',4,254,2100000),
 (7,'Summer Rain',5,201,3450000),(8,'Fiesta',5,188,4120000),
 (9,'Neon Tokyo',6,299,2670000),(10,'Bullet Train',6,276,1980000),
 (11,'Sandstorm',7,233,760000),(12,'Electric Tide',8,241,1540000),
 (13,'Harbor Lights',8,267,1120000),(14,'River Song',9,302,430000),
 (15,'Old Pine',9,288,380000),(16,'Moonrise',10,254,1890000),
 (17,'Eclipse',10,231,1660000),(18,'Midnight Drive',8,215,2010000);
INSERT INTO concert VALUES
 (1,1,'Opera House','Oslo','2023-03-12',4200),
 (2,1,'Arena Hall','Berlin','2023-05-20',7800),
 (3,2,'Stadium One','Chicago','2023-04-08',21000),
 (4,2,'The Dome','London','2023-06-15',18500),
 (5,3,'Plaza Stage','Madrid','2023-02-25',5600),
 (6,4,'Circuit Club','Tokyo','2023-07-01',3100),
 (7,4,'Neon Room','Osaka','2023-07-08',2200),
 (8,5,'Desert Fest','Marrakesh','2023-09-14',4900),
 (9,6,'Harbor Arena','Manchester','2023-08-03',9600),
 (10,7,'Pine Theater','Vancouver','2023-01-19',1400);
""",
        examples=[
            ("How many albums has each artist released?",
             "SELECT T1.artist_name, COUNT(*) FROM artist AS T1 "
             "JOIN album AS T2 ON T1.artist_id = T2.artist_id GROUP BY T1.artist_name"),
            ("List the titles of songs longer than 300 seconds.",
             "SELECT song_title FROM song WHERE duration_sec > 300"),
            ("What is the most played song?",
             "SELECT song_title FROM song ORDER BY plays DESC LIMIT 1"),
            ("Show the names of artists from Norway.",
             "SELECT artist_name FROM artist WHERE country = 'Norway'"),
            ("What is the total number of tickets sold for artist 'The Velvet Keys'?",
             "SELECT SUM(T2.tickets_sold) FROM artist AS T1 "
             "JOIN concert AS T2 ON T1.artist_id = T2.artist_id WHERE T1.artist_name = 'The Velvet Keys'"),
            ("Find albums released after 2015 along with their artist names.",
             "SELECT T2.album_title, T1.artist_name FROM artist AS T1 "
             "JOIN album AS T2 ON T1.artist_id = T2.artist_id WHERE T2.release_year > 2015"),
            ("Which genre has the most artists?",
             "SELECT genre FROM artist GROUP BY genre ORDER BY COUNT(*) DESC LIMIT 1"),
            ("List each album title and its total number of plays.",
             "SELECT T1.album_title, SUM(T2.plays) FROM album AS T1 "
             "JOIN song AS T2 ON T1.album_id = T2.album_id GROUP BY T1.album_title"),
            ("How many concerts were held in Tokyo?",
             "SELECT COUNT(*) FROM concert WHERE city = 'Tokyo'"),
            ("Show the artist name and the venue of their concerts in 2023.",
             "SELECT T1.artist_name, T2.venue FROM artist AS T1 "
             "JOIN concert AS T2 ON T1.artist_id = T2.artist_id "
             "WHERE T2.concert_date LIKE '2023%'"),
            ("What is the average song duration per album?",
             "SELECT album_id, AVG(duration_sec) FROM song GROUP BY album_id"),
            ("Find artists who have never held a concert.",
             "SELECT artist_name FROM artist WHERE artist_id NOT IN "
             "(SELECT artist_id FROM concert)"),
            ("Which albums have more than 3 songs?",
             "SELECT T1.album_title FROM album AS T1 JOIN song AS T2 ON T1.album_id = T2.album_id "
             "GROUP BY T1.album_title HAVING COUNT(*) > 3"),
            ("List song titles and the album they belong to, ordered by plays descending.",
             "SELECT T2.song_title, T1.album_title FROM album AS T1 "
             "JOIN song AS T2 ON T1.album_id = T2.album_id ORDER BY T2.plays DESC"),
            ("What is the earliest debut year among the artists?",
             "SELECT MIN(debut_year) FROM artist"),
            ("Show the total plays for each artist.",
             "SELECT T1.artist_name, SUM(T3.plays) FROM artist AS T1 "
             "JOIN album AS T2 ON T1.artist_id = T2.artist_id "
             "JOIN song AS T3 ON T2.album_id = T3.album_id GROUP BY T1.artist_name"),
            ("Find the artist with the highest total concert attendance.",
             "SELECT T1.artist_name FROM artist AS T1 "
             "JOIN concert AS T2 ON T1.artist_id = T2.artist_id "
             "GROUP BY T1.artist_name ORDER BY SUM(T2.tickets_sold) DESC LIMIT 1"),
            ("Which labels have released albums for more than one artist?",
             "SELECT label FROM album GROUP BY label HAVING COUNT(DISTINCT artist_id) > 1"),
            ("List the albums released in the same year as 'Glass Hearts'.",
             "SELECT album_title FROM album WHERE release_year = "
             "(SELECT release_year FROM album WHERE album_title = 'Glass Hearts')"),
            ("What is the number of songs per genre?",
             "SELECT T1.genre, COUNT(*) FROM artist AS T1 "
             "JOIN album AS T2 ON T1.artist_id = T2.artist_id "
             "JOIN song AS T3 ON T2.album_id = T3.album_id GROUP BY T1.genre"),
        ],
    ),
    # ------------------------------------------------------------------ #
    DbFixture(
        db_id="hospital",
        ddl="""
CREATE TABLE department (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    floor INTEGER,
    head_doctor TEXT
);
CREATE TABLE doctor (
    doctor_id INTEGER PRIMARY KEY,
    doctor_name TEXT NOT NULL,
    dept_id INTEGER,
    title TEXT,
    hire_year INTEGER,
    salary REAL,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);
CREATE TABLE patient (
    patient_id INTEGER PRIMARY KEY,
    patient_name TEXT NOT NULL,
    gender TEXT,
    birth_year INTEGER,
    city TEXT
);
CREATE TABLE visit (
    visit_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    doctor_id INTEGER,
    visit_date TEXT,
    diagnosis TEXT,
    cost REAL,
    FOREIGN KEY (patient_id) REFERENCES patient(patient_id),
    FOREIGN KEY (doctor_id) REFERENCES doctor(doctor_id)
);
CREATE TABLE prescription (
    presc_id INTEGER PRIMARY KEY,
    visit_id INTEGER,
    drug_name TEXT,
    dosage_mg INTEGER,
    days INTEGER,
    FOREIGN KEY (visit_id) REFERENCES visit(visit_id)
);
INSERT INTO department VALUES
 (1,'Cardiology',3,'Dr. Adams'),
 (2,'Neurology',4,'Dr. Brooks'),
 (3,'Orthopedics',2,'Dr. Chen'),
 (4,'Pediatrics',1,'Dr. Diaz'),
 (5,'Oncology',5,'Dr. Evans');
INSERT INTO doctor VALUES
 (1,'Alice Adams',1,'Chief',2005,210000.0),
 (2,'Brian Brooks',2,'Chief',2009,205000.0),
 (3,'Cathy Chen',3,'Chief',2011,198000.0),
 (4,'Daniel Diaz',4,'Chief',2014,190000.0),
 (5,'Eva Evans',5,'Chief',2007,215000.0),
 (6,'Frank Ford',1,'Attending',2016,165000.0),
 (7,'Grace Gray',2,'Resident',2021,95000.0),
 (8,'Henry Hall',3,'Attending',2018,158000.0),
 (9,'Iris Ito',4,'Attending',2019,152000.0),
 (10,'Jack Jones',5,'Resident',2022,92000.0);
INSERT INTO patient VALUES
 (1,'Mary Smith','F',1958,'Boston'),
 (2,'John Doe','M',1972,'Chicago'),
 (3,'Linda Park','F',1985,'Seattle'),
 (4,'Robert Lang','M',1949,'Boston'),
 (5,'Susan Reed','F',1991,'Austin'),
 (6,'Michael Tran','M',1966,'Chicago'),
 (7,'Patricia Hall','F',2001,'Denver'),
 (8,'William Scott','M',1980,'Seattle'),
 (9,'Jennifer Wu','F',1975,'Austin'),
 (10,'Charles Young','M',2010,'Denver'),
 (11,'Barbara King','F',1995,'Boston'),
 (12,'Thomas Wright','M',1963,'Chicago');
INSERT INTO visit VALUES
 (1,1,1,'2023-01-05','Hypertension',320.0),
 (2,2,1,'2023-01-11','Arrhythmia',780.0),
 (3,3,2,'2023-01-18','Migraine',450.0),
 (4,4,5,'2023-02-02','Lymphoma',4200.0),
 (5,5,3,'2023-02-14','Fracture',1100.0),
 (6,6,2,'2023-02-21','Stroke',5600.0),
 (7,7,4,'2023-03-03','Flu',180.0),
 (8,8,3,'2023-03-12','Sprain',240.0),
 (9,9,1,'2023-03-25','Hypertension',330.0),
 (10,10,4,'2023-04-02','Fever',150.0),
 (11,11,1,'2023-04-15','Chest Pain',890.0),
 (12,12,5,'2023-04-28','Anemia',1200.0),
 (13,1,1,'2023-05-10','Hypertension',300.0),
 (14,4,5,'2023-05-22','Lymphoma',3900.0),
 (15,2,1,'2023-06-01','Arrhythmia',720.0),
 (16,5,3,'2023-06-18','Fracture',980.0),
 (17,3,2,'2023-07-04','Migraine',410.0),
 (18,9,1,'2023-07-19','Chest Pain',860.0);
INSERT INTO prescription VALUES
 (1,1,'Lisinopril',10,30),(2,2,'Amiodarone',200,14),(3,3,'Sumatriptan',50,10),
 (4,4,'Rituximab',500,21),(5,5,'Ibuprofen',400,7),(6,6,'Aspirin',81,90),
 (7,7,'Oseltamivir',75,5),(8,8,'Ibuprofen',400,5),(9,9,'Lisinopril',10,30),
 (10,10,'Paracetamol',500,3),(11,11,'Nitroglycerin',5,30),(12,12,'Iron',65,60),
 (13,13,'Lisinopril',20,30),(14,14,'Rituximab',500,21),(15,15,'Amiodarone',200,14),
 (16,16,'Ibuprofen',400,7),(17,17,'Sumatriptan',50,10),(18,18,'Nitroglycerin',5,30);
""",
        examples=[
            ("How many doctors are in each department?",
             "SELECT T1.dept_name, COUNT(*) FROM department AS T1 "
             "JOIN doctor AS T2 ON T1.dept_id = T2.dept_id GROUP BY T1.dept_name"),
            ("List the names of patients who were diagnosed with Hypertension.",
             "SELECT T2.patient_name FROM visit AS T1 JOIN patient AS T2 "
             "ON T1.patient_id = T2.patient_id WHERE T1.diagnosis = 'Hypertension'"),
            ("What is the average visit cost for the Oncology department?",
             "SELECT AVG(T1.cost) FROM visit AS T1 JOIN doctor AS T2 ON T1.doctor_id = T2.doctor_id "
             "WHERE T2.dept_id = 5"),
            ("Find the doctor with the highest salary.",
             "SELECT doctor_name FROM doctor ORDER BY salary DESC LIMIT 1"),
            ("Show the total cost of visits per patient.",
             "SELECT T1.patient_name, SUM(T2.cost) FROM patient AS T1 "
             "JOIN visit AS T2 ON T1.patient_id = T2.patient_id GROUP BY T1.patient_name"),
            ("How many patients are from Boston?",
             "SELECT COUNT(*) FROM patient WHERE city = 'Boston'"),
            ("List the drug names prescribed during visits in 2023.",
             "SELECT DISTINCT T1.drug_name FROM prescription AS T1 "
             "JOIN visit AS T2 ON T1.visit_id = T2.visit_id WHERE T2.visit_date LIKE '2023%'"),
            ("Which doctors have seen more than 2 patients?",
             "SELECT T1.doctor_name FROM doctor AS T1 JOIN visit AS T2 "
             "ON T1.doctor_id = T2.doctor_id GROUP BY T1.doctor_name HAVING COUNT(*) > 2"),
            ("What is the most common diagnosis?",
             "SELECT diagnosis FROM visit GROUP BY diagnosis ORDER BY COUNT(*) DESC LIMIT 1"),
            ("Show patient names with their doctor names.",
             "SELECT T1.patient_name, T3.doctor_name FROM patient AS T1 "
             "JOIN visit AS T2 ON T1.patient_id = T2.patient_id "
             "JOIN doctor AS T3 ON T2.doctor_id = T3.doctor_id"),
            ("Find the total cost of visits with diagnosis 'Lymphoma'.",
             "SELECT SUM(cost) FROM visit WHERE diagnosis = 'Lymphoma'"),
            ("List patients born before 1960.",
             "SELECT patient_name FROM patient WHERE birth_year < 1960"),
            ("What is the number of visits per department?",
             "SELECT T1.dept_name, COUNT(*) FROM department AS T1 "
             "JOIN doctor AS T2 ON T1.dept_id = T2.dept_id "
             "JOIN visit AS T3 ON T2.doctor_id = T3.doctor_id GROUP BY T1.dept_name"),
            ("Which departments have a chief doctor who was hired before 2010?",
             "SELECT T1.dept_name FROM department AS T1 "
             "JOIN doctor AS T2 ON T1.dept_id = T2.dept_id "
             "WHERE T2.title = 'Chief' AND T2.hire_year < 2010"),
            ("Show the average dosage per drug.",
             "SELECT drug_name, AVG(dosage_mg) FROM prescription GROUP BY drug_name"),
            ("Find patients who have never visited the hospital.",
             "SELECT patient_name FROM patient WHERE patient_id NOT IN "
             "(SELECT patient_id FROM visit)"),
            ("List the names of female patients older than 50.",
             "SELECT patient_name FROM patient WHERE gender = 'F' AND 2023 - birth_year > 50"),
            ("What is the total number of prescriptions written?",
             "SELECT COUNT(*) FROM prescription"),
            ("Which doctor has the most expensive total visit cost?",
             "SELECT T1.doctor_name FROM doctor AS T1 JOIN visit AS T2 "
             "ON T1.doctor_id = T2.doctor_id GROUP BY T1.doctor_name ORDER BY SUM(T2.cost) DESC LIMIT 1"),
            ("Show the diagnosis and cost of visits that cost more than 1000.",
             "SELECT diagnosis, cost FROM visit WHERE cost > 1000"),
        ],
    ),
    # ------------------------------------------------------------------ #
    DbFixture(
        db_id="ecommerce",
        ddl="""
CREATE TABLE customer (
    cust_id INTEGER PRIMARY KEY,
    cust_name TEXT NOT NULL,
    city TEXT,
    signup_date TEXT,
    loyalty_tier TEXT
);
CREATE TABLE product (
    prod_id INTEGER PRIMARY KEY,
    prod_name TEXT NOT NULL,
    category TEXT,
    unit_price REAL,
    stock INTEGER
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    cust_id INTEGER,
    order_date TEXT,
    status TEXT,
    FOREIGN KEY (cust_id) REFERENCES customer(cust_id)
);
CREATE TABLE order_item (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER,
    prod_id INTEGER,
    quantity INTEGER,
    discount REAL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (prod_id) REFERENCES product(prod_id)
);
INSERT INTO customer VALUES
 (1,'Aaron Bell','New York','2021-03-04','gold'),
 (2,'Bella Cruz','Los Angeles','2021-07-19','silver'),
 (3,'Carl Dunn','Chicago','2022-01-22','bronze'),
 (4,'Diana Evans','New York','2022-05-30','gold'),
 (5,'Ethan Ford','Houston','2022-08-14','silver'),
 (6,'Fiona Grant','Chicago','2023-01-09','bronze'),
 (7,'George Hill','Los Angeles','2023-02-27','silver'),
 (8,'Helen Ives','New York','2023-04-18','gold');
INSERT INTO product VALUES
 (1,'Wireless Mouse','Electronics',29.99,150),
 (2,'Mechanical Keyboard','Electronics',89.99,80),
 (3,'USB-C Hub','Electronics',45.50,120),
 (4,'Standing Desk','Furniture',349.00,25),
 (5,'Ergonomic Chair','Furniture',259.00,40),
 (6,'Desk Lamp','Furniture',39.99,200),
 (7,'Notebook Set','Stationery',12.50,500),
 (8,'Gel Pen Pack','Stationery',7.99,800),
 (9,'Monitor 27in','Electronics',229.00,60),
 (10,'Bookshelf','Furniture',129.00,30);
INSERT INTO orders VALUES
 (1,1,'2023-01-05','completed'),
 (2,1,'2023-02-11','completed'),
 (3,2,'2023-01-20','completed'),
 (4,3,'2023-03-02','cancelled'),
 (5,4,'2023-02-18','completed'),
 (6,4,'2023-04-25','completed'),
 (7,5,'2023-03-15','pending'),
 (8,6,'2023-04-02','completed'),
 (9,7,'2023-05-09','completed'),
 (10,8,'2023-05-21','completed'),
 (11,2,'2023-06-03','completed'),
 (12,1,'2023-06-17','pending');
INSERT INTO order_item VALUES
 (1,1,1,2,0.0),(2,1,7,5,0.1),(3,2,2,1,0.0),(4,3,9,1,0.05),
 (5,3,3,2,0.0),(6,4,4,1,0.0),(7,5,5,2,0.1),(8,5,6,3,0.0),
 (9,6,9,1,0.15),(10,7,8,10,0.0),(11,8,1,1,0.0),(12,8,2,1,0.0),
 (13,9,10,2,0.05),(14,10,6,1,0.0),(15,10,7,4,0.0),(16,11,3,1,0.0),
 (17,12,5,1,0.0),(18,12,2,1,0.1);
""",
        examples=[
            ("How many customers are in each city?",
             "SELECT city, COUNT(*) FROM customer GROUP BY city"),
            ("List the names of products in the Electronics category.",
             "SELECT prod_name FROM product WHERE category = 'Electronics'"),
            ("What is the total revenue of all completed orders?",
             "SELECT SUM(T2.quantity * T3.unit_price * (1 - T2.discount)) FROM orders AS T1 "
             "JOIN order_item AS T2 ON T1.order_id = T2.order_id "
             "JOIN product AS T3 ON T2.prod_id = T3.prod_id WHERE T1.status = 'completed'"),
            ("Find the customer who placed the most orders.",
             "SELECT T1.cust_name FROM customer AS T1 JOIN orders AS T2 ON T1.cust_id = T2.cust_id "
             "GROUP BY T1.cust_name ORDER BY COUNT(*) DESC LIMIT 1"),
            ("Show the name and unit price of the most expensive product.",
             "SELECT prod_name, unit_price FROM product ORDER BY unit_price DESC LIMIT 1"),
            ("How many orders have the status 'pending'?",
             "SELECT COUNT(*) FROM orders WHERE status = 'pending'"),
            ("List each product category and the number of products in it.",
             "SELECT category, COUNT(*) FROM product GROUP BY category"),
            ("What is the average quantity per order item?",
             "SELECT AVG(quantity) FROM order_item"),
            ("Find the names of customers who signed up in 2023.",
             "SELECT cust_name FROM customer WHERE signup_date LIKE '2023%'"),
            ("Show the total quantity sold for each product.",
             "SELECT T1.prod_name, SUM(T2.quantity) FROM product AS T1 "
             "JOIN order_item AS T2 ON T1.prod_id = T2.prod_id GROUP BY T1.prod_name"),
            ("Which products have never been ordered?",
             "SELECT prod_name FROM product WHERE prod_id NOT IN "
             "(SELECT prod_id FROM order_item)"),
            ("List the customers with gold loyalty tier who live in New York.",
             "SELECT cust_name FROM customer WHERE loyalty_tier = 'gold' AND city = 'New York'"),
            ("What is the total amount spent by customer 'Aaron Bell'?",
             "SELECT SUM(T2.quantity * T3.unit_price * (1 - T2.discount)) FROM orders AS T1 "
             "JOIN order_item AS T2 ON T1.order_id = T2.order_id "
             "JOIN product AS T3 ON T2.prod_id = T3.prod_id WHERE T1.cust_id = 1"),
            ("How many distinct customers placed an order in 2023?",
             "SELECT COUNT(DISTINCT cust_id) FROM orders WHERE order_date LIKE '2023%'"),
            ("Show products with stock below 50.",
             "SELECT prod_name, stock FROM product WHERE stock < 50"),
            ("Which order has the highest total value?",
             "SELECT T1.order_id FROM orders AS T1 JOIN order_item AS T2 ON T1.order_id = T2.order_id "
             "JOIN product AS T3 ON T2.prod_id = T3.prod_id "
             "GROUP BY T1.order_id ORDER BY SUM(T2.quantity * T3.unit_price * (1 - T2.discount)) DESC LIMIT 1"),
            ("List the names of customers who have placed a cancelled order.",
             "SELECT DISTINCT T1.cust_name FROM customer AS T1 "
             "JOIN orders AS T2 ON T1.cust_id = T2.cust_id WHERE T2.status = 'cancelled'"),
            ("What is the average unit price of products in the Furniture category?",
             "SELECT AVG(unit_price) FROM product WHERE category = 'Furniture'"),
            ("Show each customer name and their total number of orders.",
             "SELECT T1.cust_name, COUNT(*) FROM customer AS T1 "
             "JOIN orders AS T2 ON T1.cust_id = T2.cust_id GROUP BY T1.cust_name"),
            ("Find products priced above the average product price.",
             "SELECT prod_name FROM product WHERE unit_price > (SELECT AVG(unit_price) FROM product)"),
        ],
    ),
]
