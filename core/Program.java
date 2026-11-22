import java.util.Scanner;

class Calculate {
    int add(int a, int b) {
        return a + b;
    }

    int sub(int a, int b) {
        return a - b;
    }

    int multi(int a, int b) {
        return a * b;
    }

    int div(int a, int b) {
        return a / b;
    }

    int square(int a) {
        return a * a;
    }
}

class Program {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Calculate c = new Calculate();

        System.out.print("Enter first number: ");
        int a = sc.nextInt();

        System.out.print("Enter second number: ");
        int b = sc.nextInt();

        System.out.println("Addition: " + c.add(a, b));
        System.out.println("Subtraction: " + c.sub(a, b));
        System.out.println("Multiplication: " + c.multi(a, b));
if (b == 0) { 
    System.out.println("Division: Not defined"); 
} else { 
    System.out.println("Division: " + c.div(a, b)); 
}
        System.out.println("Square of first number: " + c.square(a));
    }
}