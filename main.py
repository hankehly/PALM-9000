from palm_9000.wake_word import wait_for_wake_word


def main():
    while True:
        print("🌴 Waiting for wake word...")
        if not wait_for_wake_word():
            break


if __name__ == "__main__":
    main()
