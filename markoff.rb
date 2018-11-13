require 'numbers_in_words'

class Markoff
  def initialize
    @library = Hash.new { |h, k| h[k] = []}
  end

  # Looks at the text in chunks of two characters and records which character comes next
  def feed(text)
    add_character(:start, text.slice(0,2))
    text.length.times do |i|
      key = text.slice(i, 2)
      value = text.slice(i+2, 1)
      if value.length == 1
        add_character(key, value)
      else
        add_character(key, "end")
        break
      end
    end
  end

  # Adds the provided character the library. I'm not bothering to weight the averages in any fashion
  def add_character(a, b)
    if !@library[a].include?(b)
      @library[a].push(b)
    end
  end

  def generate()
    output = ""
    last_char = :start
    while true
      next_char = @library[last_char].sample
      if next_char == "end"
        break
      end
      output += next_char
      last_char = next_char
    end
    return output
  end

  def generate_double
    output = @library[:start].sample
    while true
      key = output.slice(-2, 2)
      value = @library[key].sample
      if value == "end"
        return output
      else
        output += value
      end
    end
  end
end

# Checks that it didn't generate a real number
def valid_number(text)
  number_version = NumbersInWords.in_numbers(text)
  back_to_words = NumbersInWords.in_words(number_version)
  return text == back_to_words
end

mark = Markoff.new()
1000.times do |i|
  power = (rand * 8).floor
  r = (rand * (10 ** power)).floor
  mark.feed(NumbersInWords.in_words(r))
end

File.open("50_Thousand_Fake_Numbers_With_Normal_Deviates.txt", 'a') do |file|
  300.times do
    word = mark.generate_double()
    if !valid_number(word)
      file.write(word+"\n")
    end
  end
end