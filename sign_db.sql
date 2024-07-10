CREATE TABLE words (
	wordNo 	int	NOT NULL auto_increment primary key,
	wordName	varchar(10)	NULL
);

CREATE TABLE handlandmark (
	handNo	int	NOT NULL auto_increment primary key,
	wordNo	int	NOT NULL,
	x	int	NULL,
	y	int	NULL,
	z	int	NULL,
    CONSTRAINT fk_handlandmark_words FOREIGN KEY (wordNo) REFERENCES words(wordNo)
);

CREATE TABLE grade (
	gradeNo	int	NOT NULL auto_increment primary key,
	wordNo	int	NOT NULL,
	levels	varchar(10)	Not NULL,
    CONSTRAINT fk_grade_words FOREIGN KEY (wordNo) REFERENCES words(wordNo)
);

CREATE TABLE mypage (
	myNo	int	NOT NULL auto_increment primary key,
	wordNo	int	NOT NULL,
	isCorrect	boolean	NULL,
	nickName	varchar(20)	NULL,
    CONSTRAINT fk_mypage_words FOREIGN KEY (wordNo) REFERENCES words(wordNo)
);

CREATE TABLE ranks (
	rankNo	int	NOT NULL auto_increment primary key,
	gradeNo	int	NOT NULL,
	myNo	int	NOT NULL,
	today	datetime	NULL,
    CONSTRAINT fk_ranks_grade FOREIGN KEY (gradeNo) REFERENCES grade(gradeNo),
    CONSTRAINT fk_ranks_mypage FOREIGN KEY (myNo) REFERENCES mypage(myNo)
);

drop table ranks;
drop table mypage;
drop table grade;
drop table handlandmark;
drop table word;


