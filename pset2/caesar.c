#include <cs50.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// prototypes
char rotate(char c, int n);
int validate_key(int argc, string argv[]);

// code
int main(int argc, string argv[])
{
    // retrieves key
    int key = validate_key(argc, argv);
    if (key == -1)
    {
        return 1; // exits the program if key is invalid
    }

    // prompts for plaintext
    string plaintext = get_string("Please enter your text: ");
    if (plaintext == NULL)
    {
        return 1; // exits the program on invalid input
    }

    // initializes ciphertext
    string ciphertext = malloc(strlen(plaintext) + 1);
    if (ciphertext == NULL)
    {
        return 1; // exits the program
    }

    // rotates each character
    for (int i = 0; plaintext[i] != '\0'; i++)
    {
        ciphertext[i] = rotate(plaintext[i], key);
    }
    ciphertext[strlen(plaintext)] = '\0';

    printf("Ciphertext: %s\n", ciphertext);
    free(ciphertext);
}

// functions
// rotates letters, not integers
char rotate(char c, int n)
{
    int norm_key = n % 26;
    if (isupper(c))
    {
        int index = c - 'A';
        c = (index + norm_key) % 26 + 'A';
    }
    else if (islower(c))
    {
        int index = c - 'a';
        c = (index + norm_key) % 26 + 'a';
    }
    return c;
}

// validates key
int validate_key(int argc, string argv[])
{
    int key = 0;

    // rejects more than 1 CLI arguments
    if (argc != 2)
    {
        printf("Usage: %s key\n", argv[0]);
        return -1; // indicates error
    }

    // confirms argv[1] is a valid digit
    bool is_valid_key = true;
    for (int i = 0; argv[1][i] != '\0'; i++)
    {
        if (!isdigit(argv[1][i]))
        {
            is_valid_key = false;
            break;
        }
    }

    // turns argv[1] to int
    if (is_valid_key)
    {
        key = atoi(argv[1]); // assigns value to key
    }
    else
    {
        printf("Usage: %s key\n", argv[0]);
        return -1; // indicates error
    }

    return key;
}
