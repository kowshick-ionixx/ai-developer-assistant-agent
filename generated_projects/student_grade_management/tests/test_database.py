from database import (
    add_mark,
    add_student,
    calculate_grade,
    get_all_students,
    get_marks_for_student,
)


def test_add_and_get_student(test_db):
    success, _ = add_student("Alice Smith", "R001", "10th Grade", db_path=test_db)
    assert success is True

    students = get_all_students(db_path=test_db)
    assert len(students) == 1
    assert students[0][1] == "Alice Smith"
    assert students[0][2] == "R001"

    # Test duplicate roll number restriction
    success_dup, _ = add_student("Bob Jones", "R001", "10th Grade", db_path=test_db)
    assert success_dup is False


def test_marks_and_grades(test_db):
    add_student("Charlie Brown", "R002", "9th Grade", db_path=test_db)
    students = get_all_students(db_path=test_db)
    student_id = students[0][0]

    add_mark(student_id, "Math", 95.0, db_path=test_db)
    add_mark(student_id, "Science", 85.0, db_path=test_db)

    marks = get_marks_for_student(student_id, db_path=test_db)
    assert len(marks) == 2

    total = sum(m[1] for m in marks)
    avg = total / len(marks)
    assert total == 180.0
    assert avg == 90.0
    assert calculate_grade(avg) == "A"
    assert calculate_grade(55.0) == "F"
